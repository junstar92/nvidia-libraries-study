/*****************************************************************************
 * File:        matrix_add2.cu
 * Description: Matrix Addition with 2D approach. A + B = C
 *              
 * Compile:     nvcc -o matrix_add matrix_add2.cu
 * Run:         ./matrix_add <bx> <by>
 *                 <bx> : the number of x dimension per a block
 *                 <by> : the number of y dimension per a block
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
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

void initMatrix(float* mat, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(float);

    int bx = 32;
    int by = 32;
    if (argc > 1) bx = strtol(argv[1], NULL, 10);
    if (argc > 2) by = strtol(argv[2], NULL, 10);

    printf("> Matrix size: %d x %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B;
    h_A = static_cast<float*>(malloc(n_bytes));
    h_B = static_cast<float*>(malloc(n_bytes));

    // init data
    initMatrix(h_A, nxy);
    initMatrix(h_B, nxy);

    // CUDA Event
    cudaEvent_t start, end;
    float msec = 0.f;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, n_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, n_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, n_bytes));

    // transfer data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(bx, by);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y -1) / block.y);
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CUDA_ERROR_CHECK(cudaEventRecord(end));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("> sumMatrixOnGPU2D<<<(%d,%d), (%d,%d)>>> (Average)Elapsted Time: %.3f msec\n", grid.x, grid.y, block.x, block.y, msec);

    // free device global memory, CUDA event
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    CUDA_ERROR_CHECK(cudaFree(d_C));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    // free host memory
    free(h_A);
    free(h_B);

    return 0;
}