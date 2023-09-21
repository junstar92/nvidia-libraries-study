/*
 * Simple wmma implementation (D = alpha * AB + beta * C).
 * - A: m x k matrix (row-major layout)
 * - B: k x n matrix (row-major layout)
 * - C,D: m x n matrix (row-major layout)
 * 
 * compile command: nvcc -o simple_wmma simple_wmma.cu -arch sm_86 (compute capability >= 72)
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

template<typename T>
void print(std::vector<T> mat, int row, int col, char const* title = nullptr)
{
    if (title != nullptr) printf("%s\n", title);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << mat[i * col + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
    typename WMMA_A_FLAG_LAYOUT, typename WMMA_B_FLAG_LAYOUT>
__global__
void mma_wmma_kernel(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    // Assume the matrices A, B, and C are row-major layout
    uint32_t const lda = k;
    uint32_t const ldb = n;
    uint32_t const ldc = n;
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
        uint32_t const a_row_idx = warp_row_idx * WMMA_M;
        uint32_t const a_col_idx = ki;
        uint32_t const b_row_idx = ki;
        uint32_t const b_col_idx = warp_col_idx * WMMA_N;

        // check bound
        if (a_row_idx < m && a_col_idx < k && b_row_idx < k && b_col_idx < n) {
            T1 const* matrix_a_mptr = A + a_row_idx * lda + a_col_idx;
            T1 const* matrix_b_mptr = B + b_row_idx * ldb + b_col_idx;

            // load the inputs
            wmma::load_matrix_sync(frag_a, matrix_a_mptr, lda);
            wmma::load_matrix_sync(frag_b, matrix_b_mptr, ldb);
            // for (int i = 0; i < warpSize; i++) {
            //     if (threadIdx.x == i) {
            //         printf("[%d] warp (%d %d) a (%d %d) b (%d %d)\n", threadIdx.x, warp_row_idx, warp_col_idx, a_row_idx, a_col_idx, b_row_idx, b_col_idx);
            //         printf("a_frag: ");
            //         for (uint32_t i = 0; i < frag_a.num_elements; i++) {
            //             printf("%3.3f ", __half2float(frag_a.x[i]));
            //         }
            //         printf("\nb_frag: ");
            //         for (uint32_t i = 0; i < frag_b.num_elements; i++) {
            //             printf("%3.3f ", __half2float(frag_b.x[i]));
            //         }
            //         printf("\n");
            //     }
            // }

            // perform the matrix multiplication
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
            // for (int i = 0; i < warpSize; i++) {
            //     if (threadIdx.x == i) {
            //         printf("[%d] ", threadIdx.x);
            //         for (uint32_t i = 0; i < frag_acc.num_elements; i++) {
            //             printf("%3.3f ", frag_acc.x[i]);
            //         }
            //         printf("\n");
            //     }
            // }
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
void mma_wmma(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, cudaStream_t stream = nullptr)
{
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
    
    mma_wmma_kernel<T1,
                    T2,
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    wmma::row_major,
                    wmma::row_major>
        <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, alpha, beta);
}

template<typename T1, typename T2>
__global__
void mma_naive(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    uint32_t const row = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t const col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        T2 sum{0};
        C[n * row + col] = 0;
        for (uint32_t i = 0; i < k; i++) {
            sum += A[k * row + i] * B[n * i + col];
        }
        C[n * row + col] = alpha * sum + beta * C[n * row + col];
    }
}

template<>
__global__
void mma_naive(half const* A, half const* B, float* C, uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    uint32_t const row = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t const col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float sum{0};
        C[n * row + col] = 0;
        for (uint32_t i = 0; i < k; i++) {
            sum += __half2float(A[k * row + i] * B[n * i + col]);
        }
        C[n * row + col] = alpha * sum + beta * C[n * row + col];
    }
}

template<typename T1, typename T2>
void mma_gpu(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, cudaStream_t stream = nullptr)
{
    float const alpha{1.f};
    float const beta{0.f};

    dim3 grid_dim, block_dim;
    block_dim.x = 32;
    block_dim.y = 16;
    grid_dim.x = (n + block_dim.x - 1) / block_dim.x;
    grid_dim.y = (m + block_dim.y - 1) / block_dim.y;

    mma_naive<<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, alpha, beta);
}

template<typename T1, typename T2>
void mma_cpu(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k)
{
    // Assume that A and B are row major layout.
    uint32_t const lda = k;
    uint32_t const ldb = n;
    uint32_t const ldc = n;

    float const alpha{1.f};
    float const beta{0.f};

    for (uint32_t mi = 0; mi < m; mi++) {
        for (int32_t ni = 0; ni < n; ni++) {
            T2 accum{0};

            for (uint32_t ki = 0; ki < k; ki++) {
                accum += A[mi * lda + ki] * B[ki * ldb + ni];
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


float get_average_error(float* C_ref, float* C, size_t n)
{
    float err{0.f};
    for (size_t i = 0; i < n; i++) {
        err += std::abs(C_ref[i] - C[i]);
    }
    return err / n;
}

int main(int argc, char** argv)
{
    uint32_t const size_m{1024};
    uint32_t const size_n{1024};
    uint32_t const size_k{1024};
    std::cout << "Matrix Size\n";
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
    float *d_matrix_a_float{nullptr}, *d_matrix_b_float{nullptr};
    float *d_matrix_c{nullptr};
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_a_half, size_m * size_k * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_b_half, size_k * size_n * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_a_float, size_m * size_k * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_b_float, size_k * size_n * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_c, size_m * size_n * sizeof(float)));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_a_half, h_matrix_a_half.data(), h_matrix_a_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_b_half, h_matrix_b_half.data(), h_matrix_b_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_a_float, h_matrix_a_float.data(), h_matrix_a_float.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_b_float, h_matrix_b_float.data(), h_matrix_b_float.size() * sizeof(float), cudaMemcpyHostToDevice));

    //print(h_matrix_a_float, size_m, size_k, "A host: ");
    //print(h_matrix_b_float, size_k, size_n, "B host: ");
    // cpu mma
    mma_cpu(h_matrix_a_float.data(), h_matrix_b_float.data(), h_matrix_c_ref.data(), size_m, size_n, size_k);
    //print(h_matrix_c_ref, size_m, size_n, "C host: ");

    std::function<void(cudaStream_t)> func_float
        {std::bind(mma_gpu<float, float>, d_matrix_a_float, d_matrix_b_float, d_matrix_c, size_m, size_n, size_k, std::placeholders::_1)};
    std::function<void(cudaStream_t)> func_half
        {std::bind(mma_gpu<half, float>, d_matrix_a_half, d_matrix_b_half, d_matrix_c, size_m, size_n, size_k, std::placeholders::_1)};
    std::function<void(cudaStream_t)> func_wmma
        {std::bind(mma_wmma<half, float>, d_matrix_a_half, d_matrix_b_half, d_matrix_c, size_m, size_n, size_k, std::placeholders::_1)};

    uint32_t const num_iters = 100;
    // mma gpu kernel
    CUDA_ERROR_CHECK(cudaMemset(d_matrix_c, 0, h_matrix_c.size() * sizeof(float)));
    float total_msec = measure_performance(func_float, stream, num_iters);
    CUDA_ERROR_CHECK(cudaMemcpy(h_matrix_c.data(), d_matrix_c, h_matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float avg_err = get_average_error(h_matrix_c_ref.data(), h_matrix_c.data(), h_matrix_c_ref.size());
    std::cout << std::fixed << std::setprecision(3)
        << "(f32f32f32) CUDA Kernel Latency : " << total_msec / num_iters << " ms (err: " << std::setprecision(6) << avg_err << ") / " 
        << std::setprecision(3) << flop / (total_msec / 1000.f / num_iters) * 1e-9 << " GFlop/s" << std::endl;

    CUDA_ERROR_CHECK(cudaMemset(d_matrix_c, 0, h_matrix_c.size() * sizeof(float)));
    total_msec = measure_performance(func_half, stream, num_iters);
    CUDA_ERROR_CHECK(cudaMemcpy(h_matrix_c.data(), d_matrix_c, h_matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    avg_err = get_average_error(h_matrix_c_ref.data(), h_matrix_c.data(), h_matrix_c_ref.size());
    std::cout << std::fixed << std::setprecision(3)
        << "(f16f16f32) CUDA Kernel Latency : " << total_msec / num_iters << " ms (err: " << std::setprecision(6) << avg_err << ") / " 
        << std::setprecision(3) << flop / (total_msec / 1000.f / num_iters) * 1e-9 << " GFlop/s" << std::endl;

    // wmma mma
    CUDA_ERROR_CHECK(cudaMemset(d_matrix_c, 0, h_matrix_c.size() * sizeof(float)));
    total_msec = measure_performance(func_wmma, stream, num_iters);
    CUDA_ERROR_CHECK(cudaMemcpy(h_matrix_c.data(), d_matrix_c, h_matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    avg_err = get_average_error(h_matrix_c_ref.data(), h_matrix_c.data(), h_matrix_c_ref.size());
    std::cout << std::fixed << std::setprecision(3)
        << "(f16f16f32) WMMA Kernel Latency : " << total_msec / num_iters << " ms (err: " << std::setprecision(6) << avg_err << ") / " 
        << std::setprecision(3) << flop / (total_msec / 1000.f / num_iters) * 1e-9 << " GFlop/s" << std::endl;
    //print(h_matrix_c, size_m, size_n, "C dev :  ");


    // free resources
    CUDA_ERROR_CHECK(cudaFree(d_matrix_a_half));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_b_half));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_a_float));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_b_float));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_c));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

    return 0;
}