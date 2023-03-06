/*****************************************************************************
 * File:        vector_add_with_streams.cu
 * Description: Implement Vector Addition with default stream and non-default streams.
 *              
 * Compile:     nvcc -O3 -o vector_add_streams vector_add_with_streams.cu
 * Run:         ./vector_add_streams
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define N 100

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

void vectorAddOnCPU(float const* a, float const* b, float* c, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__
void vectorAddOnGPU(float const* a, float const* b, float* c, int const num_elements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elements) {
        for (int i = 0; i < N; i++) {
            c[idx] = a[idx] + b[idx];
        }
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

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> Vector Addition + Data Transfer(HtoD, DtoH) at device %d: %s\n", dev, prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    int pow = 24;
    if (argc > 1) pow = strtol(argv[1], NULL, 10);
    int num_elements = 1 << pow;
    size_t bytes = num_elements * sizeof(float);

    printf("> with %d elements\n", num_elements);

    // allocate the host input vectors a, b, c
    float *h_a = (float*)malloc(sizeof(float) * num_elements);
    float *h_b = (float*)malloc(sizeof(float) * num_elements);
    float *host_ref = (float*)malloc(sizeof(float) * num_elements);
    float *gpu_ref = (float*)malloc(sizeof(float) * num_elements);

    // init vector a, b
    initVector(h_a, num_elements);
    initVector(h_b, num_elements);

    // vector addition on host
    vectorAddOnCPU(h_a, h_b, host_ref, num_elements);

    // allocate CUDA events for estimating elapsed time
    cudaEvent_t start, end;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // Example 1: vector addition with default stream
    // allocate the device input vectors a, b, c
    float *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, bytes));

    // Launch the vectorAddOnGPU
    dim3 block(256);
    dim3 grid((block.x + num_elements - 1) / block.x, 1);

    CUDA_ERROR_CHECK(cudaEventRecord(start));

    // copy(HtoD) - kernel - copy(DtoH)
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    vectorAddOnGPU<<<grid, block>>>(d_a, d_b, d_c, num_elements);
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    printf("vectorAddOnGPU with default stream         : elapsed time %f ms\n", msec);

    // check results
    checkResult(host_ref, gpu_ref, num_elements);

    free(h_a);
    free(h_b);
    free(gpu_ref);

    // Example 2: vector addition with non-default streams (4 streams)
    // allocate pinned host memory for asynchronous data transfer
    CUDA_ERROR_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_ERROR_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_ERROR_CHECK(cudaMallocHost(&gpu_ref, bytes));

    // init vector a, b
    initVector(h_a, num_elements);
    initVector(h_b, num_elements);

    // vector addition on host
    vectorAddOnCPU(h_a, h_b, host_ref, num_elements);
    
    // create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamCreate(&streams[i]));
    }

    // calculate size of data chunk that each stream processes and execution configuration
    size_t bytes_per_stream = bytes / NUM_STREAMS;
    size_t num_elements_per_stream = num_elements / NUM_STREAMS;
    grid = dim3((block.x + num_elements_per_stream - 1) / block.x, 1);

    CUDA_ERROR_CHECK(cudaEventRecord(start));

    // vectorAddOnGPU kernel launch with streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        size_t offset = i * num_elements_per_stream;
        CUDA_ERROR_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        vectorAddOnGPU<<<grid, block, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, num_elements_per_stream);
        CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_ref + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
    }
    // sync with streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    printf("vectorAddOnGPU with non-default streams(4) : elapsed time %f ms\n", msec);

    // check results
    checkResult(host_ref, gpu_ref, num_elements);

    // destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }

    // free host, device memory, CUDA event
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    CUDA_ERROR_CHECK(cudaFree(d_c));
    CUDA_ERROR_CHECK(cudaFreeHost(h_a));
    CUDA_ERROR_CHECK(cudaFreeHost(h_b));
    CUDA_ERROR_CHECK(cudaFreeHost(gpu_ref));
    free(host_ref);

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    return 0;
}