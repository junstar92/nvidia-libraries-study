/**************************************************
 * compile command: nvcc -o swizzle -O3 swizzle.cu
 **************************************************/
#include <iostream>
#include <iomanip>
#include <vector>

#include "cuda_runtime.h"

template<int BLOCK_SIZE>
__global__
void matmul_smem(float const* A, float const* B, float* C, int const M, int const N, int const K)
{
    unsigned int c_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    unsigned int c_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    __shared__ float a_smem[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_smem[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.f;
    // loop over all the sub-matrices of A and B
    // to compute the block sub-matrix
    for (unsigned int block_k = 0; block_k < K; block_k += BLOCK_SIZE) {
        // calculate row, column, data index
        a_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        a_col = block_k + threadIdx.x;
        a_idx = a_row * K + a_col;
        b_row = block_k + threadIdx.y;
        b_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        b_idx = b_row * N + b_col;

        // load the matrices from global memory to shared memory
        a_smem[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_idx] : 0.f;
        b_smem[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_idx] : 0.f;
        __syncthreads(); // synchronize to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += a_smem[threadIdx.y][i] * b_smem[i][threadIdx.x];
        }
        __syncthreads();
    }

    // write the block sub-matrix to global memory
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = acc;
    }
}

__device__
uint2 get_swizzle_idx(int const tile_size)
{
    uint2 ret;

    unsigned int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    // get strip info
    unsigned int total_cta_in_a_strip = tile_size * gridDim.y;
    unsigned int number_of_strips = gridDim.x / tile_size;
    unsigned int residual_strip_width = gridDim.x % tile_size;
    // calculate swizzle CTA ID
    unsigned int strip_id = block_idx / total_cta_in_a_strip;
    unsigned int cta_id_in_strip = block_idx % total_cta_in_a_strip;
    unsigned int use_sz = (block_idx < total_cta_in_a_strip * number_of_strips) ? tile_size : residual_strip_width;
    unsigned int strip_id_x = cta_id_in_strip % use_sz;
    unsigned int strip_id_y = cta_id_in_strip / use_sz;
    unsigned int strip_flat_idx = strip_id * tile_size + strip_id_y * gridDim.x + strip_id_x;

    ret.x = strip_flat_idx % gridDim.x;
    ret.y = strip_flat_idx / gridDim.y;

    return ret;
}

template<int BLOCK_SIZE, int STRIP_WIDTH = 4>
__global__
void matmul_swizzle(float const* A, float const* B, float* C, int const M, int const N, int const K)
{
    uint2 swizzle_idx = get_swizzle_idx(STRIP_WIDTH);

    unsigned int c_row = swizzle_idx.x;
    unsigned int c_col = swizzle_idx.y;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    __shared__ float a_smem[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_smem[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.f;
    // loop over all the sub-matrices of A and B
    // to compute the block sub-matrix
    for (unsigned int block_k = 0; block_k < K; block_k += BLOCK_SIZE) {
        // calculate row, column, data index
        a_row = BLOCK_SIZE * swizzle_idx.y + threadIdx.y;
        a_col = block_k + threadIdx.x;
        a_idx = a_row * K + a_col;
        b_row = block_k + threadIdx.y;
        b_col = BLOCK_SIZE * swizzle_idx.x + threadIdx.x;
        b_idx = b_row * N + b_col;

        // load the matrices from global memory to shared memory
        a_smem[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_idx] : 0.f;
        b_smem[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_idx] : 0.f;
        __syncthreads(); // synchronize to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += a_smem[threadIdx.y][i] * b_smem[i][threadIdx.x];
        }
        __syncthreads();
    }

    // write the block sub-matrix to global memory
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = acc;
    }
}

void check_result(float const* ref, float const* tgt, int const m, int const n)
{
    float eps = 1.e-6f;
    unsigned int idx;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            idx = n * r + c;
            if (std::abs(ref[idx] - tgt[idx]) > eps) {
                std::cout << std::fixed << std::setprecision(6)
                    << "Result verification failed at (" << r << "," << c << ") ref: " << ref[idx] << " tgt: " << tgt[idx] << "\n";
                return;
            }
        }
    }
}

int main(int argc, char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    if (argc > 1) M = std::stoi(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) K = std::stoi(argv[3]);

    std::vector<float> h_A(M * K, 1.f), h_B(K * N, 1.f), h_C(M * N, 0.f), h_Ref(M * N, 0.f);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    dim3 block{32, 32};
    dim3 grid{(N + block.x - 1) / block.x, (M + block.y - 1) / block.y};

    // baseline
    matmul_smem<32><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Ref.data(), d_C, h_Ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_C, 0, h_C.size() * sizeof(float));

    // matmul with swizzle index
    matmul_swizzle<32><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // result validation
    check_result(h_Ref.data(), h_C.data(), M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}