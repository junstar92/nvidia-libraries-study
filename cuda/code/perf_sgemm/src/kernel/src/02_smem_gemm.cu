#include <utils.hpp>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_SIZE>
__global__
void smem_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int c_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    unsigned int c_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    float acc = 0.f;
    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // calculate row, column, data index
        a_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        a_col = BLOCK_SIZE * block + threadIdx.x;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_SIZE * block + threadIdx.y;
        b_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        As[threadIdx.y][threadIdx.x] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    if (c_row < m && c_col < n) {
        C[c_row * n + c_col] = alpha * acc + beta * C[c_row * n + c_col];
    }
}

void smem_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    int constexpr block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid(cdiv(n, block.x), cdiv(m, block.y));

    smem_sgemm_kernel<block_size><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}