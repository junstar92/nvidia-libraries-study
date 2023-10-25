#include <utils.hpp>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

#define WARP_SIZE       32
#define WARP_TILE_M     16
#define WARP_TILE_N     8
#define WARP_THREAD_M   8
#define WARP_THREAD_N   4
#define THREAD_TILE_M   8
#define THREAD_TILE_N   8
#define LANE_LAYOUT     2
#define LANE_M          4
#define LANE_N          4
#define PADDING         4

template<int NUM_THREADS, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int WARP_M, int WARP_N, int WARP_K>
__global__
void warptiling_sgemm_kernel(
    int const M, int const N, int const K,
    float alpha,
    float const* A, float const* B,
    float beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    int constexpr WARP_NUM_ROW = (BLOCK_M / WARP_M);
    int constexpr WARP_NUM_COL = (BLOCK_N / WARP_N);
    int const thread_idx = cta.thread_rank();
    // threadblock-level indices
    int const block_idx_m = cta.group_index().y;
    int const block_idx_n = cta.group_index().x;
    // warp-level indices of each thread
    int const lane_idx = thread_idx % WARP_SIZE;
    int const warp_idx = thread_idx / WARP_SIZE;
    int const tb_warp_idx_m = warp_idx % (WARP_NUM_ROW);
    int const tb_warp_idx_n = warp_idx / (WARP_NUM_ROW);
    int const warp_tile_idx_m = ((lane_idx >> 3) << 1) + (lane_idx & 1);
    int const warp_tile_idx_n = ((lane_idx & 7) >> 1);
    int const tb_tile_idx_m = warp_tile_idx_m + tb_warp_idx_m * WARP_TILE_M;
    int const tb_tile_idx_n = warp_tile_idx_n + tb_warp_idx_n * WARP_TILE_N;

    // set blocktile to beginning of A's row and B's column
    A += block_idx_m * BLOCK_M * K;
    B += block_idx_n * BLOCK_N;

    // allocate smem space for the threadblock
    __shared__ float a_smem[BLOCK_K][BLOCK_M + PADDING];
    __shared__ float b_smem[BLOCK_K][BLOCK_N];

    // allocate thread-local cache for results in register file
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.f,};
    // register cache for As and Bs
    float a_frag[THREAD_TILE_M] = {0.f,};
    float b_frag[THREAD_TILE_N] = {0.f,};

    // element indices for writing smem from global memory
    int const a_tb_idx_m = thread_idx / BLOCK_K;
    int const a_tb_idx_k = thread_idx % BLOCK_K;
    int const b_tb_idx_k = thread_idx / BLOCK_N;
    int const b_tb_idx_n = thread_idx % BLOCK_N;

    // GEMM main loop - iterates over the entire K dimension - no unrolling
    for (unsigned int block_k = 0; block_k < K; block_k += BLOCK_K) {
        // load A and B tiles from global memory and store to SMEM
        #pragma unroll
        for (int k = 0; k < (BLOCK_K * BLOCK_M / NUM_THREADS); k++) {
            a_smem[a_tb_idx_k][k * (NUM_THREADS / BLOCK_K) + a_tb_idx_m] = A[(k * (NUM_THREADS / BLOCK_K) + a_tb_idx_m) * K + a_tb_idx_k];
        }
        #pragma unroll
        for (int k = 0; k < (BLOCK_K * BLOCK_N / NUM_THREADS); k++) {
            b_smem[k * (NUM_THREADS / BLOCK_N) + b_tb_idx_k][b_tb_idx_n] = B[(k * (NUM_THREADS / BLOCK_N) + b_tb_idx_k) * N + b_tb_idx_n];
        }
        cta.sync();

        // advance blocktile
        A += BLOCK_K;
        B += BLOCK_K * N;

        // warp tile structure - iterates over the thread block tile - fully unroll across BLOCK_K
        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += (BLOCK_K / WARP_K)) {
            // fetch a_frag and b_frag from SMEM corresponding to k-index
            #pragma unroll
            for (int m = 0; m < LANE_LAYOUT; m++) {
                *reinterpret_cast<float4*>(&a_frag[m * LANE_M]) = 
                    *reinterpret_cast<float4*>(&a_smem[warp_k][(tb_tile_idx_m + m * WARP_THREAD_M) * LANE_M]);
            }
            #pragma unroll
            for (int n = 0; n < LANE_LAYOUT; n++) {
                *reinterpret_cast<float4*>(&b_frag[n * LANE_N]) =
                    *reinterpret_cast<float4*>(&b_smem[warp_k][(tb_tile_idx_n + n * WARP_THREAD_N) * LANE_N]);
            }

            // mma in thread tile structure
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        cta.sync();
    }

    // set warptile to beginning of C's row and column
    C += (block_idx_m * BLOCK_M + tb_warp_idx_m * WARP_M) * N + block_idx_n * BLOCK_N + tb_warp_idx_n * WARP_N;
    // write out the results
    #pragma unroll
    for (int m = 0; m < LANE_LAYOUT; m++) {
        #pragma unroll
        for (int n = 0; n < LANE_LAYOUT; n++) {
            #pragma unroll
            for (int k = 0; k < LANE_M; k++) {
                *reinterpret_cast<float4*>(&C[((warp_tile_idx_m + m * WARP_THREAD_M) * LANE_M + k) * N + (warp_tile_idx_n + n * WARP_THREAD_N) * LANE_N]) = 
                    *reinterpret_cast<float4*>(&accum[m * LANE_M + k][n * LANE_N]);
            }
        }
    }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int WARP_K>
void warptiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    int constexpr BLOCK_WARP_M = BLOCK_M / WARP_M;
    int constexpr BLOCK_WARP_N = BLOCK_N / WARP_N;
    int constexpr NUM_THREADS_PER_BLOCK = BLOCK_WARP_M * BLOCK_WARP_N * WARP_SIZE;

    dim3 block(NUM_THREADS_PER_BLOCK);
    dim3 grid(cdiv(n, BLOCK_N), cdiv(m, BLOCK_M));
    
    warptiling_sgemm_kernel<NUM_THREADS_PER_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K>
        <<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

template void warptiling_sgemm<64, 64, 8, 64, 32, 8>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);
template void warptiling_sgemm<128, 128, 8, 64, 32, 8>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);