/*
 * Simple wmma implementation.
 * - A: m x k matrix (row-major or column-major layout)
 * - B: k x n matrix (row-major or column-major layout)
 * - C,D: m x n matrix (row-major layout)
 * 
 * 1) D = alpha * AB + beta * C (NN)
 * 2) D = alpha * AB^T + beta * C (NT)
 * 3) D = alpha * A^TB + beta * C (TN)
 * 4) D = alpha * A^TB^T + beta * C (TT)
 * 
 * compile command: nvcc -o wmma_layout wmma_layout.cu -arch sm_86 (compute capability >= 72)
 */
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <functional>

#include <cuda_runtime.h>
#include <mma.h>

#define CUDA_ERROR_CHECK(err) cuda_error_check((err), #err, __FILE__, __LINE__)

using namespace nvcuda;

void cuda_error_check(cudaError_t err, char const* const func, char const* const file, int const num_line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error(" <<  func << "): " << cudaGetErrorString(err) << " at " << file << ":" << num_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void random_init(float* arr, size_t n, std::default_random_engine& engine)
{
    std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
    for (size_t i = 0; i < n; i++) {
        arr[i] = uniform_dist(engine);
    }
}

void float2half(half* to, float const* from, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        to[i] = __float2half(from[i]);
    }
}

float get_average_error(float* C_ref, float* C, size_t n)
{
    float err{0.f};
    for (size_t i = 0; i < n; i++) {
        err += std::abs(C_ref[i] - C[i]);
    }
    return err / n;
}

template<typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
    typename WMMA_A_FLAG_LAYOUT, typename WMMA_B_FLAG_LAYOUT>
__global__
void mma_wmma_kernel(
    T1 const* A, T1 const* B, T2* C, uint32_t const m, uint32_t const n, uint32_t const k,
    uint32_t const lda, uint32_t const ldb, uint32_t const ldc, bool transA, bool transB, float alpha, float beta)
{
    // determine warp index in a 2D grid (128 x 4) 
    // => it means each block has 4x4 warp and processes 64 x 64 matrix
    uint32_t const warp_row_idx = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    uint32_t const warp_col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    // declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_A_FLAG_LAYOUT> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_B_FLAG_LAYOUT> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> frag_acc;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> frag_c;

    // initialize the output
    wmma::fill_fragment(frag_acc, static_cast<T2>(0));

    // perfor the matrix multiplication
    for (uint32_t ki = 0; ki < k; ki += WMMA_K) {
        uint32_t const a_row_idx = !transA ? warp_row_idx * WMMA_M : ki;
        uint32_t const a_col_idx = !transA ? ki : warp_row_idx * WMMA_M;
        uint32_t const b_row_idx = !transB ? ki : warp_col_idx * WMMA_N;
        uint32_t const b_col_idx = !transB ? warp_col_idx * WMMA_N : ki;

        // check bound
        if (a_row_idx < (!transA ? k : m) && a_col_idx < (!transA ? m : k)
            && b_row_idx < (!transB ? k : n) && b_col_idx < (!transB ? n : k)) {
            T1 const* matrix_a_mptr = A + a_row_idx * lda + a_col_idx;
            T1 const* matrix_b_mptr = B + b_row_idx * ldb + b_col_idx;

            // load the inputs
            wmma::load_matrix_sync(frag_a, matrix_a_mptr, lda);
            wmma::load_matrix_sync(frag_b, matrix_b_mptr, ldb);

            // perform the matrix multiplication
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }
    }

    // scale the output by beta, and add it the result scaled by alpha
    uint32_t const c_row_idx = warp_row_idx * WMMA_M;
    uint32_t const c_col_idx = warp_col_idx * WMMA_N;
    if (c_row_idx < m && c_col_idx < n) {
        T2* matrix_c_mptr = C + c_row_idx * ldc + c_col_idx;
        wmma::load_matrix_sync(frag_c, matrix_c_mptr, ldc, wmma::mem_row_major);

        for (uint32_t i = 0; i < frag_c.num_elements; i++) {
            frag_c.x[i] = alpha * frag_acc.x[i] + beta * frag_c.x[i];
        }
        // store the output
        wmma::store_matrix_sync(matrix_c_mptr, frag_c, ldc, wmma::mem_row_major);
    }
}

template<typename T1, typename T2>
void mma_wmma(T1 const* A, T1 const* B, T2* C, uint32_t const m, uint32_t const n, uint32_t const k, bool transA, bool transB, cudaStream_t stream = nullptr)
{
    uint32_t lda = !transA ? k : m;
    uint32_t ldb = !transB ? n : k;
    uint32_t ldc = n;
    float const alpha{1.f};
    float const beta{0.f};

    // shape restriction
    int const WMMA_M{16};
    int const WMMA_N{16};
    int const WMMA_K{16};

    int const warp_size{32};
    int const warp_x{4};
    int const warp_y{4};
    dim3 grid_dim, block_dim;
    // 1. each warp processes 16x16 output tile matrix
    // 2. a block processes 64x64 output tile matrix (it means there are `warp_x` x `warp_y` warps = (4x4 warps))
    // 3. consecutive threads are grouped in a warp => blockDim.x must be a multiple of warp_size(32)
    // => block_dim: (128, 4)
    // => grid_dim: (16, 16)
    block_dim.x = warp_x * warp_size;
    block_dim.y = warp_y;
    grid_dim.x = (m + (WMMA_M * warp_x - 1)) / (WMMA_M * warp_x);
    grid_dim.y = (n + (WMMA_N * warp_y - 1)) / (WMMA_N * warp_y);
    
    if (!transA && !transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::row_major,
                        wmma::row_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else if (!transA && transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::row_major,
                        wmma::col_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else if (transA && !transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::col_major,
                        wmma::row_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::col_major,
                        wmma::col_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
}

template<typename T1, typename T2>
void mma_cpu(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, bool transA, bool transB)
{
    // Assume that A and B are row major layout.
    uint32_t const lda = !transA ? k : m;
    uint32_t const ldb = !transB ? n : k;
    uint32_t const ldc = n;

    float const alpha{1.f};
    float const beta{0.f};

    for (uint32_t mi = 0; mi < m; mi++) {
        for (int32_t ni = 0; ni < n; ni++) {
            // compute C[mi][ni]
            T2 accum{0};

            for (uint32_t ki = 0; ki < k; ki++) {
                if (!transA && !transB) {
                    // accum = A[mi][ki] * B[ki][ni]
                    accum += A[mi * lda + ki] * B[ki * ldb + ni];
                }
                else if (!transA && transB) {
                    // accum = A[mi][ki] * B[ni][ki];
                    accum += A[mi * lda + ki] * B[ni * ldb + ki];
                }
                else if (transA && !transB) {
                    // accum = A[ki][mi] * B[ki][ni]
                    accum += A[ki * lda + mi] * B[ki * ldb + ni];
                }
                else {
                    // accum = A[ki][mi] * B[ni][ki]
                    accum += A[ki * lda + mi] * B[ni * ldb + ki];
                }
            }
            C[mi * ldc + ni] = alpha * accum + beta * C[mi * ldc + ni];
        }
    }
}

template<typename F>
float measure_performance(F func, cudaStream_t stream, uint32_t num_iters = 100)
{
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // warm-up
    for (uint32_t i = 0; i < 10; i++) {
        func(stream);
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    // measuer performance
    CUDA_ERROR_CHECK(cudaEventRecord(start, stream));
    for (uint32_t i = 0; i < num_iters; i++) {
        func(stream);
    }
    CUDA_ERROR_CHECK(cudaEventRecord(stop, stream));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

    float msec{};
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    return msec;
}

int main(int argc, char** argv)
{
    uint32_t const size_m{1024};
    uint32_t const size_n{1024};
    uint32_t const size_k{1024};
    std::cout << "Maxtirx Sizes\n";
    std::cout << "- M: " << size_m << std::endl
              << "- N: " << size_n << std::endl
              << "- K: " << size_k << std::endl;
    uint32_t const flop = 2 * size_m * size_n * size_k;
    
    std::default_random_engine rd_engine;

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

    // allocate host memory
    std::vector<float> h_matrix_a_float(size_m * size_k);
    std::vector<float> h_matrix_b_float(size_k * size_n);
    std::vector<half> h_matrix_a_half(size_m * size_k);
    std::vector<half> h_matrix_b_half(size_k * size_n);
    std::vector<float> h_matrix_c(size_m * size_n, 0.f);
    std::vector<float> h_matrix_c_ref(size_m * size_n, 0.f);
    // initialize host memory
    random_init(h_matrix_a_float.data(), h_matrix_a_float.size(), rd_engine);
    random_init(h_matrix_b_float.data(), h_matrix_b_float.size(), rd_engine);
    float2half(h_matrix_a_half.data(), h_matrix_a_float.data(), h_matrix_a_float.size());
    float2half(h_matrix_b_half.data(), h_matrix_b_float.data(), h_matrix_b_float.size());

    // allocate device memory
    half *d_matrix_a_half{nullptr}, *d_matrix_b_half{nullptr};
    float *d_matrix_c{nullptr};
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_a_half, size_m * size_k * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_b_half, size_k * size_n * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_c, size_m * size_n * sizeof(float)));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_a_half, h_matrix_a_half.data(), h_matrix_a_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_b_half, h_matrix_b_half.data(), h_matrix_b_half.size() * sizeof(half), cudaMemcpyHostToDevice));

    // start test
    uint32_t const num_iters = 100;
    for (bool transA : { false, true }) {
        for (bool transB : { false, true }) {
            h_matrix_c_ref.resize(size_m * size_n, 0.f);
            // cpu mma
            mma_cpu(h_matrix_a_float.data(), h_matrix_b_float.data(), h_matrix_c_ref.data(), size_m, size_n, size_k, transA, transB);

            // wmma
            std::function<void(cudaStream_t)> func_wmma
                {std::bind(mma_wmma<half, float>, d_matrix_a_half, d_matrix_b_half, d_matrix_c, size_m, size_n, size_k, transA, transB, std::placeholders::_1)};

            CUDA_ERROR_CHECK(cudaMemset(d_matrix_c, 0, h_matrix_c.size() * sizeof(float)));
            float total_msec = measure_performance(func_wmma, stream, num_iters);
            CUDA_ERROR_CHECK(cudaMemcpy(h_matrix_c.data(), d_matrix_c, h_matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float avg_err = get_average_error(h_matrix_c_ref.data(), h_matrix_c.data(), h_matrix_c_ref.size());
            std::cout << std::fixed << std::setprecision(3)
                << "(" << (!transA ? "A  " : "A^T") << " * " << (!transB ? "B  " : "B^T")
                << ") WMMA Kernel Latency : " << total_msec / num_iters << " ms (err: " << std::setprecision(6) << avg_err << ") / " 
                << std::setprecision(3) << flop / (total_msec / 1000.f / num_iters) * 1e-9 << " GFlop/s" << std::endl;
        }
    }

    // free resources
    CUDA_ERROR_CHECK(cudaFree(d_matrix_a_half));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_b_half));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_c));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

    return 0;
}