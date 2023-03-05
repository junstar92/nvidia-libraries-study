/*****************************************************************************
 * File:        blocking_and_non_blocking.cu
 * Description: This is a sample for demonstrating that the default stream blocks
 *              operations in non-default streams if non-default streams are 
 *              blocking streams and that the default stream doesn't block operations
 *              in non-default streams if non-default streams are non-blocking streams.
 * 
 *              This blocking situation can be visualized by nsight systems.
 *              
 * Compile:     nvcc -O3 -o blocking_and_non_blocking blocking_and_non_blocking.cu
 * Run:         ./blocking_and_non_blocking <flag>
 *                  <flag> : 0(cudaStreamDefault) or 1(cudaStreamNonBlocking)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define NUM_STREAMS 4

__global__
void kernel_1(int const n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__
void kernel_2(int const n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__
void kernel_3(int const n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__
void kernel_4(int const n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> At device %d: %s with num_streams=%d\n", dev, prop.name, NUM_STREAMS);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // set flag
    unsigned int flag = cudaStreamDefault;
    if (argc > 1) flag = atoi(argv[1]);

    // allocate and initialize an array of stream handles
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&streams[i], flag));
    }

    // setup execution configuration
    int num_elements = 1 << 24;
    dim3 block(512);
    dim3 grid((block.x + num_elements - 1) / block.x);

    // launch kernels with streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>(num_elements);
        kernel_2<<<grid, block, 0, streams[i]>>>(num_elements);
        kernel_3<<<grid, block>>>(num_elements);
        kernel_4<<<grid, block, 0, streams[i]>>>(num_elements);
    }

    // release all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}