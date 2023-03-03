/*****************************************************************************
 * File:        read_segment.cu
 * Description: This code is for demonstrates the impact of misaligned reads 
 *              on performance by forcing misaligned reads to occur on a float*.
 *              readOffset kernel function is launched with <<<1, 32>>> and 
 *              the number of elements is 128 (512 bytes)
 * 
 *              To check the effect of misaligned read, use 'nsight compute'
 *              with 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum' and
 *              'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct' metrics.
 *              The first metric shows the number of global memory load transaction and
 *              the second metric shows the efficiency of global memory load.
 * 
 * Compile:     nvcc -O3 -o read_segment read_segment.cu
 * Run:         ./read_segment [offset]
 *****************************************************************************/
#include <stdio.h>
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
void readOffset(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        b[i] = a[k];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> %s starting at device %d: %s\n", argv[0], dev, prop.name);

    // setup offset and block size
    int offset = 0;
    int block_size = 32;
    if (argc > 1) offset = atoi(argv[1]);

    // setup vector size
    int num_elements = 128;
    size_t bytes = num_elements * sizeof(float);
    printf("> Vector Size: %d / %ld bytes\n", num_elements, bytes);

    // setup execution configuration
    dim3 block(block_size);
    dim3 grid = 1;

    // allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);

    // initialize host vector
    initVector(h_a, num_elements);

    // allocate device memory
    float *d_a, *d_b;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, bytes));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // CUDA event for estimating elapsed time
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    float msec;

    // kernel
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    readOffset<<<1, block>>>(d_a, d_b, num_elements, offset);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f ms\n", grid.x, block.x, offset, msec);

    // free host and device memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}