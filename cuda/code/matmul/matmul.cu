/*****************************************************************************
 * File:        matmul.cu
 * Description: Implement matrix multiplications (C = AB) using 
 *              global memory and shared memory(tiling approach).
 *              
 *              Matrix Dimensions
 *              - A: m x k
 *              - B: k x n
 *              - C: m x n
 *              
 * Compile:     nvcc -O3 -o matmul matmul.cu
 * Run:         ./matmul [m] [k] [n] [block_size]
 *                  [m]: the number of row dimension in A matrix (default: 1024)
 *                  [k]: the number of column dimensions in matrix A (default: 1024)
 *                       or the number of row dimensions in matrix B
 *                  [n]: the number of column dimensions in matrix B (default: 1024)
 *         [block_size]: the number of x and y dimensions in a thread block
 *                       It must to be set to 16 or 32(default)
 *****************************************************************************/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

__global__
void matmulNaive(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.f;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += A[k * row + i] * B[n * i + col];
        }
        C[n * row + col] = sum;
    }
}

template<int BLOCK_SIZE>
__global__
void matmulSmem(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int c_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    unsigned int c_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    float sum = 0.f;
    // loop over all the sub-matrices of A and B
    // to compute the block sub-matrix
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

        // calculate row, column, data index
        a_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        a_col = BLOCK_SIZE * block + threadIdx.x;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_SIZE * block + threadIdx.y;
        b_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        Asub[threadIdx.y][threadIdx.x] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bsub[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // synchronize to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    if (c_row < m && c_col < n) {
        C[c_row * n + c_col] = sum;
    }
}

void initMatrix(float* mat, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        mat[i] = (rand() & 0xFF) / 1.e3;
    }
}

void matmulHost(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            float sum = 0.f;
            for (int i = 0; i < k; i++) {
                sum += A[k * r + i] * B[n * i + c];
            }
            C[n * r + c] = sum;
        }
    }
}

void checkResult(float const* host_ref, float const* gpu_ref, int const m, int const n)
{
    float eps = 1.e-5f;
    unsigned int idx;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            idx = n * r + c;
            if (fabsf(host_ref[idx] - gpu_ref[idx]) > eps) {
                std::cout << std::fixed << std::setprecision(6)
                    << "Result verification failed at (" << r << "," << c << ") host: " << host_ref[idx] << " gpu: " << gpu_ref[idx] << "\n";
                return;
            }
        }
    }
}

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp dev_prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&dev_prop, dev));
    std::cout << "> Starting matrix multiplication at device " <<  dev << ": " << dev_prop.name << "\n";
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // matrix size
    int m = 1024;
    int k = 1024;
    int n = 1024;
    if (argc > 1) m = strtol(argv[1], nullptr, 10);
    if (argc > 2) k = strtol(argv[2], nullptr, 10);
    if (argc > 3) n = strtol(argv[3], nullptr, 10);

    std::cout << "> Matrix A  : (" << m << " x " << k << ")\n"
        << "> Matrix B  : (" << k << " x " << n << ")\n";

    // block size
    int block_size = 32;
    if (argc > 4) block_size = strtol(argv[4], nullptr, 10);
    std::cout << "> BLOCK_SIZE: " << block_size << "\n";

    // allocate host memory
    float *h_A, *h_B, *host_ref, *gpu_ref;
    h_A = new float[m * k];
    h_B = new float[k * n];
    host_ref = new float[m * n];
    gpu_ref = new float[m * n];

    initMatrix(h_A, m * k);
    initMatrix(h_B, k * n);
    std::fill(host_ref, host_ref + m * n, 0.f);

    // cuda event
    cudaEvent_t start, stop;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // matmul on host
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    matmulHost(h_A, h_B, host_ref, m, k, n);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    std::cout << "matmulHost\t: " << msec << " ms\n";

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    // copy matrix A,B from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // set execution configuration
    dim3 block(block_size, block_size);
    dim3 grid((block.x + n - 1) / block.x, (block.y + m - 1) / block.y);
    
    // naive matmul kernel
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    std::cout << "matmulNaive\t: " << msec << " ms\n";

    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkResult(host_ref, gpu_ref, m, n);

    std::fill(gpu_ref, gpu_ref + m * n, 0.f);
    CUDA_ERROR_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));

    // matmul kernel using shared memory
    CUDA_ERROR_CHECK(cudaEventRecord(start));
    if (block_size == 16) {
        matmulSmem<16><<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    }
    else if (block_size == 32) {
        matmulSmem<32><<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    std::cout << "matmulSmem\t: " << msec << " ms\n";

    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ref, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkResult(host_ref, gpu_ref, m, n);


    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    CUDA_ERROR_CHECK(cudaFree(d_C));
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    delete[] h_A;
    delete[] h_B;
    delete[] host_ref;
    delete[] gpu_ref;


    return 0;
}