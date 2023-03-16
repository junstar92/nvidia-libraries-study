/*****************************************************************************
 * File:        simple_shfl.cu
 * Description: This sample demonstrates a variety of shuffle instructions.
 *                  - 
 *              
 * Compile:     nvcc -O3 -o simple_shfl simple_shfl.cu
 * Run:         ./simple_shfl
 *****************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define BDIMX   16
#define SEGM    4

void printData(int* data, int const size)
{
    for (int i = 0; i < size; i++) {
        printf("%2d ", data[i]);
    }
    printf("\n");
}

__global__
void test_shfl_broadcast(int* d_out, int* d_in, int const src_lane)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff, value, src_lane, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__
void test_shfl_wrap(int* d_out, int* d_in, int const offset)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff, value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__
void test_shfl_up(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_up_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__
void test_shfl_down(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_down_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__
void test_shfl_xor(int* d_out, int* d_in, int const mask)
{
    int value = d_in[threadIdx.x];
    value = __shfl_xor_sync(0xffffffff, value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__
void test_shfl_xor_array(int* d_out, int* d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[0] = __shfl_xor_sync(0xffffffff, value[0], mask, BDIMX);
    value[1] = __shfl_xor_sync(0xffffffff, value[1], mask, BDIMX);
    value[2] = __shfl_xor_sync(0xffffffff, value[2], mask, BDIMX);
    value[3] = __shfl_xor_sync(0xffffffff, value[3], mask, BDIMX);

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

__inline__ __device__
void swap(int* value, int lane_idx, int mask, int first_idx, int second_idx)
{
    bool pred = ((lane_idx / mask + 1) % 2 == 1);

    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }

    value[second_idx] = __shfl_xor_sync(0xffffffff, value[second_idx], mask, BDIMX);

    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }
}

__global__
void test_shfl_swap(int* d_out, int* d_in, int const mask, int first_idx, int second_idx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, first_idx, second_idx);

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp dev_prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&dev_prop, dev));
    printf("> Starting at device %d: %s\n", dev, dev_prop.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    int num_elements = BDIMX;
    int h_in[BDIMX], h_out[BDIMX];

    for (int i = 0; i < num_elements; i++)
        h_in[i] = i;

    size_t bytes = num_elements * sizeof(int);
    int *d_in, *d_out;
    CUDA_ERROR_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_out, bytes));
    
    CUDA_ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BDIMX);

    // shuffle broadcast
    test_shfl_broadcast<<<1, block>>>(d_out, d_in, 2);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle broadcast\t: ");
    printData(h_out, num_elements);
    printf("\n");

    // shuffle offset (left, right)
    test_shfl_wrap<<<1, block>>>(d_out, d_in, 2);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle wrap left\t: ");
    printData(h_out, num_elements);
    printf("\n");

    test_shfl_wrap<<<1, block>>>(d_out, d_in, -2);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle wrap right\t: ");
    printData(h_out, num_elements);
    printf("\n");

    // shuffle up
    test_shfl_up<<<1, block>>>(d_out, d_in, 2);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle up\t\t: ");
    printData(h_out, num_elements);
    printf("\n");

    // shuffle down
    test_shfl_down<<<1, block>>>(d_out, d_in, 2);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle down\t\t: ");
    printData(h_out, num_elements);
    printf("\n");

    // shuffle xor
    test_shfl_xor<<<1, block>>>(d_out, d_in, 1);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle xor 1\t\t: ");
    printData(h_out, num_elements);
    printf("\n");

    test_shfl_xor<<<1, block>>>(d_out, d_in, -1);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle xor -1\t\t: ");
    printData(h_out, num_elements);
    printf("\n");
    
    // shuffle xor - array
    test_shfl_xor_array<<<1, block.x / SEGM>>>(d_out, d_in, 1);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle xor array 1\t: ");
    printData(h_out, num_elements);
    printf("\n");

    // shuffle xor swap
    test_shfl_swap<<<1, block.x / SEGM>>>(d_out, d_in, 1, 0, 3);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("initial data\t\t: ");
    printData(h_in, num_elements);
    printf("shuffle xor swap\t: ");
    printData(h_out, num_elements);
    printf("\n");

    CUDA_ERROR_CHECK(cudaFree(d_in));
    CUDA_ERROR_CHECK(cudaFree(d_out));
    CUDA_ERROR_CHECK(cudaDeviceReset());

    return 0;
}