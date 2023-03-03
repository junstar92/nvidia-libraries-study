/*****************************************************************************
 * File:        read_segment2.cu
 * Description: This code is for demonstrates the impact of unrolling techniques 
 *              on performance.
 * 
 *              To check the effect of unrolling technique, use 'nsight compute'
 *              with 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
 *              'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
 *              'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum' and
 *              'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct' metrics.
 *              These metrics show the number of global memory load/store transaction and
 *              the efficiency of global memory load/store.
 * 
 * Compile:     nvcc -O3 -o read_segment2 read_segment2.cu
 * Run:         ./read_segment2 [offset] [block_size] [power of 2 for the number of elements]
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
void warmup(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        b[i] = a[k];
}

__global__
void readOffset(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        b[i] = a[k];
}

__global__
void readOffsetUnroll4(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n) {
        b[i] = a[k];
        b[i + blockDim.x] = a[k + blockDim.x];
        b[i + 2 * blockDim.x] = a[k + 2 * blockDim.x];
        b[i + 3 * blockDim.x] = a[k + 3 * blockDim.x];
    }
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
    int block_size = 256;
    if (argc > 1) offset = atoi(argv[1]);
    if (argc > 2) block_size = atoi(argv[2]);

    // setup vector size
    int pow = 24;
    if (argc > 3) pow = atoi(argv[3]);
    int num_elements = (1 << pow);
    size_t bytes = num_elements * sizeof(float);
    printf("> Vector Size: %d / %ld bytes\n", num_elements, bytes);

    // setup execution configuration
    dim3 block(block_size);
    dim3 grid((num_elements + block.x - 1) / block.x);

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

    // kernel: warmup
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    warmup<<<grid, block>>>(d_a, d_b, num_elements, offset);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("warmup            <<< %5d, %5d >>> offset %5d elapsed %f ms\n", grid.x, block.x, offset, msec);

    // kernel: readOffset
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    readOffset<<<grid, block>>>(d_a, d_b, num_elements, offset);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("readOffset        <<< %5d, %5d >>> offset %5d elapsed %f ms\n", grid.x, block.x, offset, msec);

    // kernel: readOffsetUnroll4
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_a, d_b, num_elements, offset);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("readOffsetUnroll4 <<< %5d, %5d >>> offset %5d elapsed %f ms\n", grid.x / 4, block.x, offset, msec);

    // free host and device memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}