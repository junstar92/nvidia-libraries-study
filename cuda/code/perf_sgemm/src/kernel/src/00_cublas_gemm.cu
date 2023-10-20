#include <cublas.h>

void cublas_sgemm(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C,
    cudaStream_t stream,
    cublasHandle_t handle
)
{
    cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, A, k, B, n, &beta, C, n);
}