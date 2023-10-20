#pragma once
#include <cuda_runtime.h>

void naive_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream = nullptr
);

void smem_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream = nullptr
);

void smem_1d_blocktiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
);

void smem_2d_blocktiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
);

template<int BM, int BN, int BK>
void vectorize_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
);

template<int THREADS_PER_BLOCK, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N,
    int WARP_M = 32, int WARP_N = 16
>
void warptiling_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream
);

void cublas_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream,
    cublasHandle_t handle
);