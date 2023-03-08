/*****************************************************************************
 * File:        async_api.cu
 * Description: This example demonstrates how to control asynchronous work launched
 *              on the GPU by using events. In this example, asynchronous copies and
 *              an asynchronous kernel are used. A CUDA event is used to be determine
 *              when that work has completed
 *              
 * Compile:     nvcc -O3 -o async_api async_api.cu
 * Run:         ./async_api
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

__global__
void kernel(float* inout, float const val)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    inout[idx] = inout[idx] + val;
}

void checkResult(float* data, int const n, float const x)
{
    for (int i = 0; i < n; i++) {
        if (data[i] != x) {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return;
        }
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> At device %d: %s\n", dev, prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    int num_elements = 1 << 24;
    size_t bytes = num_elements * sizeof(int);
    float val = 10.f;

    // allocate host memory
    float* h_data;
    CUDA_ERROR_CHECK(cudaMallocHost(&h_data, bytes));
    (void)memset(h_data, 0, bytes);

    // allocate device memory
    float* d_data;
    CUDA_ERROR_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_ERROR_CHECK(cudaMemset(d_data, 255, bytes));

    // setup execution configuration
    dim3 block(512);
    dim3 grid((num_elements + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // asynchronous issue work to the GPU (all to default stream)
    CUDA_ERROR_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_data, val);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaEventRecord(stop));

    // have CPU do some work while waiting for GPU works to finish
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // print the cpu counter
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check results
    checkResult(h_data, num_elements, val);

    // release resource
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    CUDA_ERROR_CHECK(cudaFreeHost(h_data));
    CUDA_ERROR_CHECK(cudaFree(d_data));

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}