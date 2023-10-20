#include <utils.hpp>
#include <cooperative_groups.h>
#include <iostream>
namespace cg = cooperative_groups;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K,
        int THREAD_M, int THREAD_N,
        int WARP_M, int WARP_N
>
__global__
void warptiling_sgemm_kernel(
    int const m, int const n, int const k,
    float alpha,
    float const* A, float const* B,
    float beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    __shared__ float a_smem[BLOCK_K * BLOCK_M];
    __shared__ float b_smem[BLOCK_K * BLOCK_N];
    float a_frag[THREAD_M] = {0.f}, b_frag[THREAD_N] = {0.f}, acc[THREAD_M][THREAD_N] = {0.f};

    // thread, block, warp, and lane identication
    unsigned int const tid = cta.thread_rank();
    unsigned int const bx = cta.group_index().x;
    unsigned int const by = cta.group_index().y;

    // set blocktile to beginning of A's row, B's column, and C
    A += by * BLOCK_M * k;
    B += bx * BLOCK_N;
    C += by * BLOCK_M * n + bx * BLOCK_N;

    // calculate the indices that this thread will load into SMEM
    // - load 32 bytes => 4 elements per thread at each step
    unsigned int const a_inner_row = tid / (BLOCK_K / 4);
    unsigned int const a_inner_col = tid % (BLOCK_K / 4);
    unsigned int const a_row_offset = cta.num_threads() / 2 / (BLOCK_K / 4);
    unsigned int const b_inner_row = (tid - cta.num_threads() / 2) / (BLOCK_N / 4);
    unsigned int const b_inner_col = (tid - cta.num_threads() / 2) % (BLOCK_N / 4);
    unsigned int const b_row_offset = cta.num_threads() / 2 / (BLOCK_N / 4);

    unsigned int a_idx, b_idx;
    if (cta.num_threads() == 64) {
        a_idx = ((tid >> 1) & 7);
        b_idx = (((tid & 0x30) >> 3) | (tid & 1));
    }
    else if (cta.num_threads() == 256) {
        a_idx = (((tid & 128) >> 4) | ((tid >> 1) & 7));
        b_idx = (((tid & 0x70) >> 3) | (tid & 1));
    }

    // GEMM main loop - iterates over the entire K dimensions - no unrolling
    for (int block_k = 0; block_k < k; block_k += BLOCK_K) {
        // load A and B matrics from global memory and store to SMEM
        if (tid < cta.num_threads() / 2) {
            #pragma unroll
            for (unsigned int offset = 0; offset < BLOCK_M; offset += a_row_offset) {
                float4 tmp = *(float4 const*)(&A[(a_inner_row + offset) * k + a_inner_col * 4]);
                a_smem[(a_inner_col * 4 + 0) * BLOCK_M + a_inner_row + offset] = tmp.x;
                a_smem[(a_inner_col * 4 + 1) * BLOCK_M + a_inner_row + offset] = tmp.y;
                a_smem[(a_inner_col * 4 + 2) * BLOCK_M + a_inner_row + offset] = tmp.z;
                a_smem[(a_inner_col * 4 + 3) * BLOCK_M + a_inner_row + offset] = tmp.w;
            }
        }
        else {
            #pragma unroll
            for (unsigned int offset = 0; offset < BLOCK_K; offset += b_row_offset) {
                *(float4*)(&b_smem[(b_inner_row + offset) * BLOCK_N + b_inner_col * 4]) =
                    *(float4 const*)(&B[(b_inner_row + offset) * n + b_inner_col * 4]);
            }
        }
        cta.sync();

        // advance blocktile
        A += BLOCK_K;
        B += BLOCK_K * n;

        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k++) {
            // fetch a_frag and b_frag from SMEM corresponding to k-index
            *(float4*)(a_frag) = *(float4*)(&a_smem[warp_k * BLOCK_M + a_idx * 4]);
            *(float4*)(b_frag) = *(float4*)(&b_smem[warp_k * BLOCK_N + b_idx * 4]);

            #pragma unroll
            for (unsigned int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (unsigned int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        cta.sync();
    }
    
    // write out results
    unsigned int const c_row = a_idx * 4;
    unsigned int const c_col = b_idx * 4;
    #pragma unroll
    for (unsigned int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (unsigned int j = 0; j < THREAD_N; j++) {
            C[(c_row + i) * n + c_col + j] = alpha * acc[i][j] + beta * C[(c_row + i) * n + c_col + j];
        }
    }
}

template<int THREADS_PER_BLOCK, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
    int WARP_M = 32, int WARP_N = 16>
void warptiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    static_assert(BLOCK_M == BLOCK_N);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(cdiv(n, BLOCK_N), cdiv(m, BLOCK_M));

    // size_t smem_sz = sizeof(float) * BLOCK_K * BLOCK_N * 2;
    //CUDA_ERROR_CHECK(cudaFuncSetAttribute(&warptiling_sgemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_sz)));
    
    warptiling_sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N, WARP_M, WARP_N><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

template void warptiling_sgemm<64, 32, 32, 8, 4, 4>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);
template void warptiling_sgemm<256, 64, 64, 8, 4, 4>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);