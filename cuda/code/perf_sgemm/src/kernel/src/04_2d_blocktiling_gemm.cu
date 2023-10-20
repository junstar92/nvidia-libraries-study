#include <utils.hpp>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__
void smem_2d_blocktiling_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int const n_blocks = (k + BLOCK_K - 1) / BLOCK_K;
    unsigned int const thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int const a_thread_row = thread_idx / BLOCK_K;
    unsigned int const a_thread_col = thread_idx % BLOCK_K;
    unsigned int const b_thread_row = thread_idx / (BLOCK_N / THREAD_N);
    unsigned int const b_thread_col = thread_idx % (BLOCK_N / THREAD_N);
    unsigned int row, col, offset;

    float Tacc[THREAD_M][THREAD_N] = {0.f,};
    float At[THREAD_M] = {0.f,};
    float Bt[THREAD_N] = {0.f,};

    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    #pragma unroll
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_M][BLOCK_K];
        __shared__ float Bs[BLOCK_K][BLOCK_N];

        // load the matrices from global memory to shared memory
        offset = BLOCK_M / THREAD_M; // row offset for A
        #pragma unroll
        for (unsigned int o = 0; o < BLOCK_M; o += offset) {
            row = BLOCK_M * blockIdx.y + a_thread_row + o;
            col = BLOCK_K * block + a_thread_col;
            As[a_thread_row + o][a_thread_col] = (row < m && col < k) ? A[row * k + col] : 0.f;
        }
        offset = BLOCK_N / THREAD_N; // col offset for B
        #pragma unroll
        for (unsigned int o = 0; o < BLOCK_N; o += offset) {
            row = BLOCK_K * block + b_thread_row;
            col = BLOCK_N * blockIdx.x + b_thread_col + o;
            Bs[b_thread_row][b_thread_col + o] = (row < k && col < n) ? B[row * n + col] : 0.f;
        }
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (unsigned int i = 0; i < BLOCK_K; i++) { // dot product loop
            // block into registers
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_M; t++) {
                At[t] = As[a_thread_row * THREAD_M + t][i];
            }
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_N; t++) {
                Bt[t] = Bs[i][b_thread_col * THREAD_N + t];
            }

            #pragma unroll
            for (unsigned int tm = 0; tm < THREAD_M; tm++) {
                #pragma unroll
                for (unsigned int tn = 0; tn < THREAD_N; tn++) {
                    Tacc[tm][tn] += At[tm] * Bt[tn];
                }
            }
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    #pragma unroll
    for (unsigned int tm = 0; tm < THREAD_M; tm++) {
        #pragma unroll
        for (unsigned int tn = 0; tn < THREAD_N; tn++) {
            row = BLOCK_M * blockIdx.y + THREAD_M * threadIdx.y + tm;
            col = BLOCK_N * blockIdx.x + THREAD_N * threadIdx.x + tn;
            if (row < m && col < n) {
                C[row * n + col] = alpha * Tacc[tm][tn] + beta * C[row * n + col];
            }
        }
    }
}

void smem_2d_blocktiling_sgemm(
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
    int constexpr THREAD_N = BLOCK_K;
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    dim3 grid(cdiv(n, BLOCK_N), cdiv(m, BLOCK_M));
    
    smem_2d_blocktiling_sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}