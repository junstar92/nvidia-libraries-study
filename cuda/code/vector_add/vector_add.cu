/*****************************************************************************
 * File:        vector_add.cu
 * Description: Parallel Vector Addition. A + B = C
 *              
 * Compile:     nvcc -o vector_add vector_add.cu
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

__global__
void vectorAddKernel(float const* a, float const* b, float* c, int const num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements)
        c[i] = a[i] + b[i];
}

void vectorAdd(float const* a, float const* b, float* c, int const num_elements)
{
    // allocate the device input vectors a, b, c
    float *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, sizeof(float) * num_elements));

    // copy the host input vector a and b in host memory
    // to the device input vectors in device memory
    printf("> Copy input data from the host memory to the CUDA device\n");
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, a, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, b, sizeof(float) * num_elements, cudaMemcpyHostToDevice));

    // allocate CUDA events for estimating elapsed time
    cudaEvent_t start, stop;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // Launch the vectorAddKernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    printf("> CUDA kernel launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);

    CUDA_ERROR_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));

    // copy the device result vector in device memory
    // to the host result vector in host memory
    printf("> Copy output data from the CUDA device to the host memory\n");
    CUDA_ERROR_CHECK(cudaMemcpy(c, d_c, sizeof(float) * num_elements, cudaMemcpyDeviceToHost));

    // verify that the result vector is correct
    printf("> Verifying vector addition...\n");
    for (int i = 0; i < num_elements; i++) {
        if (a[i] + b[i] != c[i]) {
            fprintf(stderr, "Result verification failed at element %d (%f != %f)\n", i, a[i]+b[i], c[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("> Test PASSED\n");

    // compute performance
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    double flops = static_cast<double>(num_elements);
    double giga_flops = (flops * 1.0e-9f) / (msec / 1000.f);
    printf("Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f ops, "
            " WorkgroundSize = %u threads/block\n",
            giga_flops, msec, flops, threads_per_block);
    
    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    CUDA_ERROR_CHECK(cudaFree(d_c));
    // free CUDA event
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv)
{
    int pow = strtol(argv[1], NULL, 10);
    int num_elements = 1 << pow;

    printf("[Vector addition of %d elements on GPU]\n", num_elements);

    // allocate the host input vectors a, b, c
    float *a = (float*)malloc(sizeof(float) * num_elements);
    float *b = (float*)malloc(sizeof(float) * num_elements);
    float *c = (float*)malloc(sizeof(float) * num_elements);

    // init vector a, b
    initVector(a, num_elements);
    initVector(b, num_elements);

    // call vectorAdd function
    vectorAdd(a, b, c, num_elements);

    // free host memory
    free(a);
    free(b);
    free(c);

    printf("Done\n");

    return 0;
}