#include <utils.hpp>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__
void vectorize_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    unsigned int const num_threads = BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N);
    assert(num_threads == blockDim.x);

    // row, col index in block
    unsigned int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
    unsigned int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

    // set blocktile to beginning of A's row and B's column
    A += blockIdx.y * BLOCK_M * k;
    B += blockIdx.x * BLOCK_N;
    C += blockIdx.y * BLOCK_M * n + blockIdx.x * BLOCK_N;

    // calculate the indices that this thread will load into SMEM:
    // - load 128bit => 4 elements per thread at each step
    unsigned int const a_inner_row = threadIdx.x / (BLOCK_K / 4);
    unsigned int const a_inner_col = threadIdx.x % (BLOCK_K / 4);
    unsigned int const b_inner_row = threadIdx.x / (BLOCK_N / 4);
    unsigned int const b_inner_col = threadIdx.x % (BLOCK_N / 4);
    unsigned int const a_stride = num_threads / (BLOCK_K / 4);
    unsigned int const b_stride = num_threads / (BLOCK_N / 4);

    // allocate thread-local cache for results in register file
    float Tacc[THREAD_M * THREAD_N] = {0.f,};
    // register cache for As and Bs
    float At[THREAD_M] = {0.f,};
    float Bt[THREAD_N] = {0.f,};

    // outer loop over block tiles
    #pragma unroll
    for (unsigned int bk_idx = 0; bk_idx < k; bk_idx += BLOCK_K) {
        // allocate smem space for the current blocktile
        __shared__ float As[BLOCK_M * BLOCK_K];
        __shared__ float Bs[BLOCK_K * BLOCK_N];

        // load the matrices from global memory to shared memory
        // transpose A at this point
        #pragma unroll
        for (unsigned int offset = 0; offset < BLOCK_M; offset += a_stride) {
            float4 tmp = *reinterpret_cast<float4 const*>(&A[(a_inner_row + offset) * k + a_inner_col * 4]);
            As[(a_inner_col * 4 + 0) * BLOCK_M + a_inner_row + offset] = tmp.x;
            As[(a_inner_col * 4 + 1) * BLOCK_M + a_inner_row + offset] = tmp.y;
            As[(a_inner_col * 4 + 2) * BLOCK_M + a_inner_row + offset] = tmp.z;
            As[(a_inner_col * 4 + 3) * BLOCK_M + a_inner_row + offset] = tmp.w;
        }
        #pragma unroll
        for (unsigned int offset = 0; offset < BLOCK_K; offset += b_stride) {
            *reinterpret_cast<float4*>(&Bs[(b_inner_row + offset) * BLOCK_N + b_inner_col * 4]) =
                *reinterpret_cast<float4 const*>(&B[(b_inner_row + offset) * n + b_inner_col * 4]);
        }
        cta.sync(); // sync to make sure the matrices are loaded

        // advance blocktile
        A += BLOCK_K;       // move blocktile to right
        B += BLOCK_K * n;   // move blocktile to down

        // calculate per-thread results
        #pragma unroll
        for (unsigned int i = 0; i < BLOCK_K; i++) { // dot product loop
            // block into registers
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_M; t++) {
                At[t] = As[i * BLOCK_M + thread_row * THREAD_M + t];
            }
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_N; t++) {
                Bt[t] = Bs[i * BLOCK_N + thread_col * THREAD_N + t];
            }

            #pragma unroll
            for (unsigned int tm = 0; tm < THREAD_M; tm++) {
                #pragma unroll
                for (unsigned int tn = 0; tn < THREAD_N; tn++) {
                    Tacc[THREAD_N * tm + tn] += At[tm] * Bt[tn];
                }
            }
        }
        cta.sync();
    }

    // write out the results
    #pragma unroll
    for (unsigned int tm = 0; tm < THREAD_M; tm++) {
        #pragma unroll
        for (unsigned int tn = 0; tn < THREAD_N; tn += 4) {
            // load C vector into registers
            float4 tmp = *reinterpret_cast<float4*>(&C[(thread_row * THREAD_M + tm) * n + thread_col * THREAD_N + tn]);
            // perform GEMM update in reg
            tmp.x = alpha * Tacc[tm * THREAD_N + tn] + beta * tmp.x;
            tmp.y = alpha * Tacc[tm * THREAD_N + tn + 1] + beta * tmp.y;
            tmp.z = alpha * Tacc[tm * THREAD_N + tn + 2] + beta * tmp.z;
            tmp.w = alpha * Tacc[tm * THREAD_N + tn + 3] + beta * tmp.w;
            // write back
            *reinterpret_cast<float4*>(&C[(thread_row * THREAD_M + tm) * n + thread_col * THREAD_N + tn]) = tmp;
        }
    }
}

template<int BM, int BN, int BK>
void vectorize_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    int constexpr BLOCK_M = BM;
    int constexpr BLOCK_N = BN;
    int constexpr BLOCK_K = BK;
    int constexpr THREAD_M = BLOCK_K;
    int constexpr THREAD_N = BLOCK_K;
    dim3 block(BLOCK_M * BLOCK_N / THREAD_M / THREAD_N);
    dim3 grid(cdiv(n, BLOCK_N), cdiv(m, BLOCK_M));
    
    vectorize_sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

template void vectorize_sgemm<64, 64, 8>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);
template void vectorize_sgemm<128, 128, 8>(int const, int const, int const, float const, float const*, float const*, float const, float*, cudaStream_t);