/*****************************************************************************
 * File:        test_AoS.cu
 * Description: This code is for studying the impact on performance of
 *              data layout(AoS) on the GPU.
 * 
 *              To check the performance of data layout, use 'nsight compute'
 *              with 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct' and
 *              'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` metrics.
 *              The first metrics shows the global memory load efficiency and
 *              the second metrics shows the global memory store efficiency.
 * 
 * Compile:     nvcc -O3 -o test_AoS test_AoS.cu
 * Run:         ./test_AoS
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define NUM_ELEMENTS (1 << 24)

struct innerStruct {
    float x;
    float y;
};

void initInnerStruct(innerStruct* data, int const n)
{
    for (int i = 0; i < n; i++) {
        data[i].x = static_cast<float>((rand() & 0xFF) / 100.f);
        data[i].y = static_cast<float>((rand() & 0xFF) / 100.f);
    }
}

__global__
void testInnerStruct(innerStruct* data, innerStruct* result, int const n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> %s: test struct of array at device %d: %s\n", argv[0], dev, prop.name);

    // allocate host memory
    size_t bytes = NUM_ELEMENTS * sizeof(innerStruct);
    innerStruct* h_A = (innerStruct*)malloc(bytes);

    // initialize host array
    initInnerStruct(h_A, NUM_ELEMENTS);

    // allocate device memory
    innerStruct *d_A, *d_B;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, bytes));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // execution configuration
    int blocksize = 256;
    if (argc > 1) blocksize = atoi(argv[1]);
    dim3 block(blocksize);
    dim3 grid((NUM_ELEMENTS + block.x - 1) / block.x);

    // CUDA event for estimating elapsed time
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    float msec;

    // kernel
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    testInnerStruct<<<grid, block>>>(d_A, d_B, NUM_ELEMENTS);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    printf("testInnerStruct <<< %3d, %3d >>> elapsed %f ms\n", grid.x, block.x, msec);

    // free memory
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    free(h_A);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}