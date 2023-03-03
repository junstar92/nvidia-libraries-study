/*****************************************************************************
 * File:        matrix_add.cu
 * Description: Matrix Addition with several grid/thread sizes. A + B = C
 *              
 * Compile:     nvcc -o matrix_add matrix_add.cu
 * Run:         ./matrix_add <n> <bx> <by> <i>
 *                  <n> : specify the matrix dimension as (2^n, 2^n)
 *                 <bx> : the thread number of x dimension per a block
 *                 <by> : the thread number of y dimension per a block
 *                  <i> : 
 *                      - 0: 2D grid with 2D blocks
 *                      - 1: 1D grid with 1D blocks
 *                      - 2: 2D grid with 1D blocks
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

void sumMatrixOnHost(float const* A, float const* B, float* C, int const nx, int const ny)
{
    float const* ia = A;
    float const* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

__global__
void sumMatrixOnGPU2D(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__
void sumMatrixOnGPU1D(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx) {
        int idx;
        for (int iy = 0; iy < ny; iy++) {
            idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__
void sumMatrixOnGPUMix(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

void initMatrix(float* mat, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    int pow = strtol(argv[1], NULL, 10);
    int nx = 1 << pow;
    int ny = 1 << pow;
    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(float);

    int bx = strtol(argv[2], NULL, 10);
    int by = strtol(argv[3], NULL, 10);
    int kernel = strtol(argv[4], NULL, 10);

    printf("> Matrix size: %d x %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *host_ref, *gpu_ref;
    h_A = static_cast<float*>(malloc(n_bytes));
    h_B = static_cast<float*>(malloc(n_bytes));
    host_ref = static_cast<float*>(malloc(n_bytes));
    gpu_ref = static_cast<float*>(malloc(n_bytes));

    // init data
    initMatrix(h_A, nxy);
    initMatrix(h_B, nxy);
    (void*)memset(host_ref, 0, n_bytes);
    (void*)memset(gpu_ref, 0, n_bytes);

    // CUDA Event
    cudaEvent_t start, end;
    float msec, total = 0.f;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // add matrix at host side for result checks
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    sumMatrixOnHost(h_A, h_B, host_ref, nx, ny);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("> sumMatrixOnHost  Elapsted Time: %.3f msec\n", msec);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, n_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, n_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, n_bytes));

    // transfer data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice));

    // select kernel and execution configuration
    char* kernel_name;
    void(*sumMatrixOnGPU)(float const*, float const*, float*, int const, int const);
    dim3 block, grid;
    if (kernel == 0) {
        block = dim3(bx, by);
        grid = dim3((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        sumMatrixOnGPU = &sumMatrixOnGPU2D;
        kernel_name = "sumMatrixOnGPU2D";
    }
    else if (kernel == 1) {
        block = dim3(bx);
        grid = dim3((nx + block.x - 1) / block.x);
        sumMatrixOnGPU = &sumMatrixOnGPU1D;
        kernel_name = "sumMatrixOnGPU1D";
    }
    else {
        block = dim3(bx);
        grid = dim3((nx + block.x - 1) / block.x, ny);
        sumMatrixOnGPU = &sumMatrixOnGPUMix;
        kernel_name = "sumMatrixOnGPUMix";
    }

    // launch kernel
    for (int i = 0; i < 100; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
        CUDA_ERROR_CHECK(cudaEventRecord(end));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
        total += msec;
    }
    printf("> %s<<<(%d,%d), (%d,%d)>>> (Average)Elapsted Time: %.3f msec\n", kernel_name, grid.x, grid.y, block.x, block.y, total / 100);

    // copy kernel result back to host side
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_C, n_bytes, cudaMemcpyDeviceToHost));

    // check device results
    printf("> Verifying vector addition...\n");
    for (int i = 0; i < nxy; i++) {
        if (host_ref[i] != gpu_ref[i]) {
            fprintf(stderr, "Result verification failed at element %d (%f != %f)\n", i, host_ref[i], gpu_ref[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("> Test PASSED\n");

    // free device global memory, CUDA event
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    CUDA_ERROR_CHECK(cudaFree(d_C));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    // free host memory
    free(h_A);
    free(h_B);
    free(host_ref);
    free(gpu_ref);

    return 0;
}