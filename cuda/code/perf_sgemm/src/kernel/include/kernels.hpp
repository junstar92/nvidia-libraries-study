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

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int WARP_K>
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