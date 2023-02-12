/*****************************************************************************
 * File:        reduce_integer.cu
 * Description: Implement kernel functions for reduction(sum) problem
 *                  - recursiveReduce
 *                  - reduceNeighbored
 *                  - reduceNeighboredLess
 *                  - reduceInterleaved
 *              
 * Compile:     nvcc -o reduce_integer reduce_integer.cu
 * Run:         ./reduce_integer <n>
 *                 <n> : block size (1D). (default: 512)
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

// recursive implementation of interleaved pair approach
int recursiveReduce(int* data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// neighbored pair implementation with branch divergence
__global__
void reduceNeighbored(int* g_in, int* g_out, unsigned int const n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            in[tid] += in[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

// neighbored pair implementation with less divergence
__global__
void reduceNeighboredLess(int* g_in, int* g_out, unsigned int const n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    int index;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        index = 2 * stride * tid;

        if (index < blockDim.x)
            in[index] += in[index + stride];

        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

// interleaved pair implementation with less divergence
__global__
void reduceInterleaved(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            in[tid] += in[tid + stride];
        
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp dev_prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&dev_prop, dev));
    printf("> Starting reduction at device %d: %s\n", dev, dev_prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // array size
    int num_elements = 1 << 24;
    printf("> Array size: %d\n", num_elements);

    // execution configuration
    int block_size = 512;
    if (argc > 1) block_size = atoi(argv[1]);

    dim3 block(block_size);
    dim3 grid((num_elements + block.x - 1) / block.x);
    printf("> grid %d  block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = num_elements * sizeof(int);
    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // init the input array
    for (int i = 0; i < num_elements; i++)
        h_in[i] = (int)(rand() & 0xFF);
    (void*)memcpy(tmp, h_in, bytes);

    // cuda event
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // allocate device memory
    int *d_in, *d_out;
    CUDA_ERROR_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_out, grid.x * sizeof(int)));

    float msec = 0.f;
    int gpu_sum = 0;
    
    // cpu reduction
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    int cpu_sum = recursiveReduce(tmp, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("cpu reduce              elapsed %.4f ms    cpu sum: %d\n", msec, cpu_sum);

    // kernel 1: reduceNeighbored
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceNeighbored<<<grid, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("gpu Neighbored          elapsed %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighboredLess
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceNeighboredLess<<<grid, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("gpu NeighboredLess      elapsed %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    reduceInterleaved<<<grid, block>>>(d_in, d_out, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));

    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_out[i];
    printf("gpu reduceInterleaved   elapsed %.4f ms     gpu sum: %d <<<grid %d block %d>>>\n", msec, gpu_sum, grid.x, block.x);


    // free host memory
    free(h_in);
    free(h_out);
    free(tmp);

    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_in));
    CUDA_ERROR_CHECK(cudaFree(d_out));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    // reset device
    CUDA_ERROR_CHECK(cudaDeviceReset());

    return 0;
}