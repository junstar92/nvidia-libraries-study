/*****************************************************************************
 * File:        matrix_add_managed.cu
 * Description: Matrix addition example using CUDA mamanged memory (unified memory).
 *              CUDA will automatically manage the transfer of data to and from
 *              the GPU as needed by the application. There is no need for the
 *              programmer to use cudaMemcpy, cudaHostGetDevicePointer, or any other
 *              CUDA API.
 *              
 * Compile:     nvcc -O3 matrix_add_managed.cu -o managed
 * Run:         ./managed <n>
 *                  <n> : specify matrix dimension as (2^n, 2^n)
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

void initMatrix(float* in, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        in[i] = rand() / (float)RAND_MAX;
    }
}

void matrixAddOnHost(float const* A, float const* B, float* C, int const nx, int const ny)
{
    float const* ia = A;
    float const* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkResult(float const* host_ref, float const* gpu_ref, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        if (host_ref[i] != gpu_ref[i]) {
            printf("different on %dth element, host: %f / gpu: %f\n", i, host_ref[i], gpu_ref[i]);
            break;
        }
    }
}

__global__
void warmup(float const* A, float const* B, float* C, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__
void matrixAddGPU(float const* A, float const* B, float* C, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> Matrix Addition(Manual) at device %d: %s\n", dev, prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // setup matrix dimensions (2048 x 2048)
    int pow = 12;
    if (argc > 1) pow = atoi(argv[1]);
    int nx = 1 << pow;
    int ny = 1 << pow;

    printf("> with matrix %d x %d\n", nx, ny);
    size_t bytes = nx * ny * sizeof(float);

    // CUDA event to estimate elapsed time
    cudaEvent_t start, end;
    float msec = 0.f;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // malloc host memory
    float *A, *B, *host_ref, *gpu_ref;
    CUDA_ERROR_CHECK(cudaMallocManaged(&A, bytes));
    CUDA_ERROR_CHECK(cudaMallocManaged(&B, bytes));
    CUDA_ERROR_CHECK(cudaMallocManaged(&host_ref, bytes));
    CUDA_ERROR_CHECK(cudaMallocManaged(&gpu_ref, bytes));

    // initialize data at host side
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    initMatrix(A, nx * ny);
    initMatrix(B, nx * ny);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("initialization: \t %f sec\n", msec * 1e-3);

    memset(host_ref, 0, bytes);
    memset(gpu_ref, 0, bytes);

    // matrix addition at host side for checking result
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    matrixAddOnHost(A, B, host_ref, nx, ny);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("matrixAdd on host:\t %f sec\n", msec * 1e-3);


    // setup execution configuration
    int blockx = 32;
    int blocky = 32;
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel with unified memory all pages will migrate from host to device
    warmup<<<grid, block>>>(A, B, gpu_ref, nx, ny);

    // kernel launch
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    matrixAddGPU<<<grid, block>>>(A, B, gpu_ref, nx, ny);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("matrixAdd on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", msec * 1e-3, grid.x, grid.y, block.x, block.y);

    // check kernel error
    CUDA_ERROR_CHECK(cudaGetLastError());

    // check results
    checkResult(host_ref, gpu_ref, nx * ny);

    // free host & device memory
    CUDA_ERROR_CHECK(cudaFree(A));
    CUDA_ERROR_CHECK(cudaFree(B));
    CUDA_ERROR_CHECK(cudaFree(host_ref));
    CUDA_ERROR_CHECK(cudaFree(gpu_ref));

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}