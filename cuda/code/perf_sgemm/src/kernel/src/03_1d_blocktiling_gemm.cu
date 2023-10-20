#include <utils.hpp>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M>
__global__
void smem_1d_blocktiling_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_K - 1) / BLOCK_K;
    unsigned int const thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int const a_thread_row = thread_idx / BLOCK_K;
    unsigned int const a_thread_col = thread_idx % BLOCK_K;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx, c_row, c_col;

    float Tacc[THREAD_M] = {0.f, };

    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    #pragma unroll
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_M][BLOCK_K];
        __shared__ float Bs[BLOCK_K][BLOCK_N];
        // calculate row, column, data index of A & B
        a_row = BLOCK_M * blockIdx.y + a_thread_row;
        a_col = BLOCK_K * block + a_thread_col;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_K * block + threadIdx.y;
        b_col = BLOCK_N * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        As[a_thread_row][a_thread_col] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i++) { // dot product loop
            // we make dot product loop, which facilitates
            // resue of the Bs element, which we can cache in a tmp register.
            float Btmp = Bs[i][threadIdx.x];
            #pragma unroll
            for (int t = 0; t < THREAD_M; t++) { // inner loop
                Tacc[t] += As[threadIdx.y + THREAD_M * t][i] * Btmp;
            }
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    #pragma unroll
    for (int t = 0; t < THREAD_M; t++) {
        c_row = BLOCK_M * blockIdx.y + t * THREAD_M + threadIdx.y;
        c_col = BLOCK_N * blockIdx.x + threadIdx.x;
        if (c_row < m && c_col < n) {
            C[c_row * n + c_col] = alpha * Tacc[t] + beta * C[c_row * n + c_col];
        }
    }
}

void smem_1d_blocktiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    int constexpr BLOCK_M = 64;
    int constexpr BLOCK_N = 64;
    int constexpr BLOCK_K = 8;
    int constexpr THREAD_M = BLOCK_K;
    dim3 block(BLOCK_N, BLOCK_M / THREAD_M);
    dim3 grid(cdiv(n, BLOCK_N), cdiv(m, BLOCK_M));
    
    smem_1d_blocktiling_sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}