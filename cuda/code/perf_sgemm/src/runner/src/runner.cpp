#include <random>
#include <iostream>
#include <iomanip>

#include <runner.hpp>
#include <timer.hpp>
#include <kernels.hpp>
#include <cuda_utils.hpp>

void random_init(float* arr, size_t const n, std::default_random_engine& engine)
{
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (size_t i = 0; i < n; i++) {
        arr[i] = dist(engine);
    }
}

template<typename F>
float perf(F&& func, Runner::Result& result, int const num_warmup = 25, int const num_iteration = 100, cudaStream_t stream = nullptr)
{
    // warmup
    for (int i = 0; i < num_warmup; i++) {
        func(stream);
    }

    // perf
    Timer timer;
    for (int i = 0; i < num_iteration; i++) {
        timer.tic(stream);
        func(stream);
        timer.toc(stream);
    }

    // get results
    result.max = timer.max();
    result.min = timer.min();
    result.med = timer.med();
    result.avg = timer.avg();

    return timer.avg();
}

inline float calc_bandwidth(int const m, int const n, int const k, float const ms)
{
    long const total = 3L * m * n * k * sizeof(float) + 1L * m * n * sizeof(float);
    return (total / std::pow(2, 30)) / (ms / std::pow(2, 10));
}

inline float calc_tflops(int const m, int const n, int const k, float const ms)
{
    long const flops = 2L * m * n * k + 3L * m * n;
    return (flops / std::pow(2, 40)) / (ms / std::pow(2, 10));
}

inline bool is_allclose(float const* a, float const* b, size_t const n, float const rtol = 1e-5f, float const atol=1e-3f)
{
    for (int i = 0; i < n; i++) {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) return false;
    }

    return true;
}

Runner::Runner(std::vector<int> const& sizes, int num_warmup, int num_iteration) : sizes{sizes}, num_warmup{num_warmup}, num_iteration{num_iteration} {}

void Runner::init()
{
    cuda_device_info(0);

    kernels = {
        Kernel("cuBLAS", sizes),
        Kernel("naive_sgemm", sizes),
        Kernel("smem_sgemm", sizes),
        Kernel("smem_1d_blocktiling_sgemm<64, 64, 8, 8>", sizes),
        Kernel("smem_2d_blocktiling_sgemm<64, 64, 8, 8>", sizes),
        Kernel("vectorize_sgemm<64, 64, 8, 8>", sizes),
        Kernel("vectorize_sgemm<128, 128, 8, 8>", sizes),
        Kernel("warptiling_sgemm_kernel<64, 32, 32, 8, 4, 4>", sizes),
        Kernel("warptiling_sgemm_kernel<256, 64, 64, 8, 4, 4>", sizes),
    };
    num_kernels = kernels.size();
}

void Runner::run()
{
    for (size_t i = 0; i < sizes.size(); i++) {
        run_kernels(i);
    }
}

void Runner::print_results() const
{
    std::ostringstream oss;
    
    oss << "\n- Results (ms)\n";
    oss << std::setw(50) << "Size (M=N=K)";
    for (auto const size : sizes) {
        oss << std::setw(8) << size;
    }
    oss << "\n"
        << std::setfill('-') << std::setw(50 + 8 * sizes.size()) << "-" << std::setfill(' ') <<  "\n";
    auto write_row = [&oss](Kernel const& kernel, std::vector<int> const& sizes) {
        oss << std::setw(50) << kernel.kernel_name;
        for (int i = 0; i < sizes.size(); i++) {
            oss << std::setprecision(3) << std::fixed << std::setw(8) << kernel.results[i].avg;
        }
        oss << "\n";
    };
    for (auto const& kernel : kernels) {
        write_row(kernel, sizes);
    }
    oss << "\n";

    std::cout << oss.str();
}

void Runner::run_kernels(int test_idx)
{
    int const size = sizes[test_idx];
    std::cout << "Test " << test_idx + 1 << " (M = N = K = " << size << ")\n";
    int M, N, K;
    float alpha{1.f}, beta{0.f};
    M = N = K = size;

    // host memory allocation
    std::vector<float> h_A, h_B, h_C, ref_C;
    h_A.resize(M * K);
    h_B.resize(K * N);
    h_C.resize(M * N);
    ref_C.resize(M * N);

    // init data
    std::default_random_engine rd_engine;
    random_init(h_A.data(), h_A.size(), rd_engine);
    random_init(h_B.data(), h_B.size(), rd_engine);

    // device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // memcpy from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    // baseline (cuBLAS)
    cublasHandle_t handle;
    CUBLAS_ERROR_CHECK(cublasCreate_v2(&handle));

    std::function<void(cudaStream_t)> func;
    for (int i = 0; i < num_kernels; i++) {
        switch (i)
        {
        case 0:
            func = std::bind(cublas_sgemm, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1, handle);
            break;

        case 1:
            func = std::bind(naive_sgemm, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 2:
            func = std::bind(smem_sgemm, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 3:
            func = std::bind(smem_1d_blocktiling_sgemm, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 4:
            func = std::bind(smem_2d_blocktiling_sgemm, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 5:
            func = std::bind(vectorize_sgemm<64, 64, 8>, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 6:
            func = std::bind(vectorize_sgemm<128, 128, 8>, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 7:
            func = std::bind(warptiling_sgemm<64, 32, 32, 8, 4, 4>, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;
        
        case 8:
            func = std::bind(warptiling_sgemm<256, 64, 64, 8, 4, 4>, M, N, K, alpha, d_A, d_B, beta, d_C, std::placeholders::_1);
            break;

        default:
            break;
        }

        float avg = perf(func, kernels[i].results[test_idx], num_warmup, num_iteration);
        std::cout << " - " << std::setw(50) << std::left << kernels[i].kernel_name << ": " << std::setprecision(3) << std::fixed << std::setw(6) << avg << " ms ("
            << calc_tflops(M, N, K, avg) <<  " TFLOPs/s)";
        
        if (i == 0) {
            std::cout << "\n";
            // ref result
            CUDA_ERROR_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

            // transpose reference result
            for (int i = 0; i < h_C.size(); i++) {
                int row = i / N;
                int col = i % N;
                ref_C[col * N + row] = h_C[row * N + col];
            }
        }
        else {
            CUDA_ERROR_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (is_allclose(h_C.data(), ref_C.data(), h_C.size())) {
                std::cout << " ... PASS\n";
            }
            else {
                std::cout << " ... FAIL\n";
            }
        }
        CUDA_ERROR_CHECK(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));
    }

    // free cublas resource
    cublasDestroy_v2(handle);

    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
    CUDA_ERROR_CHECK(cudaFree(d_C));
}