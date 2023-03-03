/*****************************************************************************
 * File:        vector_add_zerocopy.cu
 * Description: Parallel Vector Addition with Zero-copy Memory. A + B = C
 *              
 * Compile:     nvcc -O3 -o vector_add_zerocopy vector_add_zerocopy.cu
 * Run:         ./vector_add <n>
 *                  <n> : the number of vector elements = n power(s) of 2
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

void initVector(float* vec, const int num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        vec[i] = rand() / (float)RAND_MAX;
    }
}

void vectorAddOnHost(float const* a, float const* b, float* c, const int num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__
void vectorAddKernel(float const* a, float const* b, float* c, int const num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements)
        c[i] = a[i] + b[i];
}

void checkResult(float const* host_ref, float const* gpu_ref, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        if (host_ref[i] != gpu_ref[i]) {
            printf("Result verification failed at element %d (%f != %f)\n", i, host_ref[i], gpu_ref[i]);
            return;
        }
    }
}

int main(int argc, char** argv)
{
    int pow = 22;
    if (argc > 1) pow = strtol(argv[1], NULL, 10);
    int num_elements = 1 << pow;
    size_t bytes = num_elements * sizeof(float);

    if (pow < 18) {
        printf("> Vector size: %d elements bytes: %3.0f KB\n", num_elements, (float)bytes / (1024.f));
    }
    else {
        printf("> Vector size: %d elements bytes: %3.0f MB\n", num_elements, (float)bytes / (1024.f*1024.f));
    }

    cudaEvent_t start, stop;
    float msec, total = 0.f;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    /*************** Case 1: using device memory ***************/
    // allocate the host memory
    float *h_a, *h_b, *host_ref, *gpu_ref;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    host_ref = (float*)malloc(bytes);
    gpu_ref = (float*)malloc(bytes);

    // init vector a, b
    initVector(h_a, num_elements);
    initVector(h_b, num_elements);
    (void*)memset(host_ref, 0, bytes);
    (void*)memset(gpu_ref, 0, bytes);
    
    // add vector at host side for result check
    vectorAddOnHost(h_a, h_b, host_ref, num_elements);

    // malloc device global memory
    float *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, bytes));

    // transfer data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // setup execution configuration
    int threads_per_block = 512;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // launch kernel at host side
    for (int i = 0; i < 100; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
        total += msec;
    }

    // copy kernel result back to host side
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_c, bytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(host_ref, gpu_ref, num_elements);

    // free device global memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    printf("> vectorAddKernel(global memory)    Elapsed Time: %f ms\n", total / 100);

    /*************** Case 2: using zero-copy memory ***************/
    total = 0.f;
    // allocate zero-copy memory
    unsigned int flags = cudaHostAllocMapped;
    CUDA_ERROR_CHECK(cudaHostAlloc(&h_a, bytes, flags));
    CUDA_ERROR_CHECK(cudaHostAlloc(&h_b, bytes, flags));

    // initialize data at host side
    initVector(h_a, num_elements);
    initVector(h_b, num_elements);
    (void*)memset(host_ref, 0, bytes);
    (void*)memset(gpu_ref, 0, bytes);
    
    // add vector at host side for result check
    vectorAddOnHost(h_a, h_b, host_ref, num_elements);

    // pass the pointer to device
    CUDA_ERROR_CHECK(cudaHostGetDevicePointer(&d_a, h_a, 0));
    CUDA_ERROR_CHECK(cudaHostGetDevicePointer(&d_b, h_b, 0));

    // launch kernel with zero-copy memory
    for (int i = 0; i < 100; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
        total += msec;
    }

    // copy kernel result back to host side
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_c, bytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(host_ref, gpu_ref, num_elements);

    // free memory
    CUDA_ERROR_CHECK(cudaFree(d_c));
    CUDA_ERROR_CHECK(cudaFreeHost(h_a));
    CUDA_ERROR_CHECK(cudaFreeHost(h_b));
    free(host_ref);
    free(gpu_ref);

    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("> vectorAddKernel(zero-copy memory) Elapsed Time: %f ms\n", total / 100);

    cudaDeviceReset();
    printf("Done\n");

    return 0;
}