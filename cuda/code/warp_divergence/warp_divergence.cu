/*****************************************************************************
 * File:        warp_divergence.cu
 * Description: Test for warp divergence. *              
 * Compile:     nvcc -o warp_divergence -G warp_divergence.cu
 * Run:         ./warp_divergence
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

__global__
void warmingup(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    a = b = 0.f;
    
    if ((tid / warpSize) % 2 == 0) {
    	a = 100.0f;
    }
    else {
    	b = 200.0f;
    }
    c[tid] = a + b;
}

__global__
void mathKernel1(float* c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0.f, b = 0.f;

    if (tid % 2 == 0) {
        a = 100.f;
    }
    else {
        b = 200.f;
    }
    c[tid] = a + b;
}

__global__
void mathKernel2(float* c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0.f, b = 0.f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.f;
    }
    else {
        b = 200.f;
    }
    c[tid] = a + b;
}


int main(int argc, char** argv)
{
    // set up data size
    int pow = 6;
    int block = 64;
    if (argc > 1) block = atoi(argv[1]);
    if (argc > 2) pow = atoi(argv[2]);
    int size = 1 << pow;
    printf("> Data size : %d\n", size);

    // set up execution configuration
    int grid = (size + block - 1) / block;
    printf("Execution Configure (block %d, grid %d)\n", block, grid);

    // allocate gpu memory
    float *d_C;
    size_t n_bytes = size * sizeof(float);
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, n_bytes));

    // CUDA event for timing kernel execution
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    float msec = 0.f;

    // warming up
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    warmingup<<<grid, block>>>(d_C);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("warmingup  <<< %4d %4d >>> elapsed time: %f msec \n", grid, block, msec);

    // run kernel 1
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    mathKernel1<<<grid, block>>>(d_C);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("mathKernel1<<< %4d %4d >>> elapsed time: %f msec \n", grid, block, msec);

    // run kernel 2
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    mathKernel2<<<grid, block>>>(d_C);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("mathKernel2<<< %4d %4d >>> elapsed time: %f msec \n", grid, block, msec);

    // free device memory and CUDA event
    CUDA_ERROR_CHECK(cudaFree(d_C));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    return 0;
}