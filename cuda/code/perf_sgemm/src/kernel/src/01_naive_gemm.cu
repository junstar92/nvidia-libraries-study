#include <utils.hpp>

__global__
void naive_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x; // col
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y; // row

    // bound check
    if (y < m && x < n) {
        float acc = 0.f;
        for (unsigned int i = 0; i < k; i++) {
            acc += A[y * k + i] * B[i * n + x];
        }
        C[y * n + x] = alpha * acc + beta * C[y * n + x];
    }
}

void naive_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
)
{
    dim3 block(32, 32);
    dim3 grid(cdiv(n, block.x), cdiv(m, block.y));
    naive_sgemm_kernel<<<grid, block, 0 , stream>>>(m, n, k, alpha, A, B, beta, C);
}