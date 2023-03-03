/*****************************************************************************
 * File:        cmem_stencil.cu
 * Description: This is example of using constant memory to optimize performance
 *              of a stencil computation by storing coefficients of the function
 *              in a constant memory array
 *              
 * Compile:     nvcc -O3 -o cmem_stencil cmem_stencil.cu
 * Run:         ./cmem_stencil
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define RADIUS 4
#define BDIM 32

// constant memory
__constant__ float coef[RADIUS + 1];

// FD coefficient
#define a0  0.00000f
#define a1  0.80000f
#define a2 -0.20000f
#define a3  0.03809f
#define a4 -0.00357f

void initData(float* in, int const num_elements)
{
    for (int i = 0; i < num_elements + RADIUS; i++) {
        in[i] = (float)(rand() & 0xFF) / 100.f;
    }
}

void printData(float* in, int const num_elements)
{
    for (int i = RADIUS; i < num_elements; i++) {
        printf("%f ", in[i]);
    }
    printf("\n");
}

void setupCoefConstant()
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

void stencil1DCPU(float* in, float* out, int const num_elements)
{
    for (int i = RADIUS; i <= num_elements; i++) {
        float tmp = a1 * (in[i+1] - in[i-1])
                    + a2 * (in[i+2] - in[i-2])
                    + a3 * (in[i+3] - in[i-3])
                    + a4 * (in[i+4] - in[i-4]);
        out[i] = tmp;
    }
}

void checkResult(float const* host_ref, float const* gpu_ref, int const num_elements)
{
    double epsilon = 1.0e-6;

    for (int i = RADIUS; i < num_elements; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, host_ref[i], gpu_ref[i]);
            break;
        }
    }
}

__global__
void stencil1DGPU(float* in, float* out, int const n)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < n) {
        // index to shared memory for stencil calculation
        int sidx = threadIdx.x + RADIUS;

        // read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS) {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        __syncthreads(); // sync to ensure all the data is available

        // apply the stencil
        float tmp = 0.f;

        #pragma unroll
        for (int i = 1; i <= RADIUS; i++) {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }
        // store the result
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> Stencil 1D at device %d: %s\n", dev, prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // set up data size
    int num_elements = 1 << 24;

    printf("> with array size: %d\n", num_elements);
    size_t bytes = (num_elements + 2 * RADIUS) * sizeof(float);

    // allocate host memory
    float* h_in = (float*)malloc(bytes);
    float* host_ref = (float*)malloc(bytes);
    float* gpu_ref = (float*)malloc(bytes);

    // allocate device memory
    float* d_in, *d_out;
    CUDA_ERROR_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_out, bytes));

    // initialize host array
    initData(h_in, num_elements + 2 * RADIUS);

    // copy to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // setup constant memory (coefficient)
    setupCoefConstant();

    // execution configuration
    dim3 block(BDIM, 1);
    dim3 grid((prop.maxGridSize[0] < num_elements / block.x)
                                    ? prop.maxGridSize[0]
                                    : num_elements / block.x, 1);
    
    // CUDA event to estimate elapsed time
    cudaEvent_t start, end;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // launch stencil1DGPU() kernel
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    stencil1DGPU<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    printf("stencil1DGPU <<< %4d, %4d >>> elapsed time: %f ms\n", grid.x, block.x, msec);

    // coyp result back to host
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_out, bytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    stencil1DCPU(h_in, host_ref, num_elements);

    // check results
    checkResult(host_ref, gpu_ref, num_elements);

    // cleanup
    CUDA_ERROR_CHECK(cudaFree(d_in));
    CUDA_ERROR_CHECK(cudaFree(d_out));
    free(h_in); free(host_ref); free(gpu_ref);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}