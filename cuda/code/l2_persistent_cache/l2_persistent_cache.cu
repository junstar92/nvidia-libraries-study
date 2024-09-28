// compile command: nvcc -o l2-persistent-cache l2_persistent_cache.cu -arch compute_80
#include <iostream>
#include <vector>
#include <iomanip>
#include <functional>
#include <cuda_runtime_api.h>

__global__
void reset_data(int* streaming_data, int const* persistent_data_lut, size_t const num_elem_streaming, size_t const num_elem_persistent)
{
    size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elem_streaming; i += stride) {
        streaming_data[i] = persistent_data_lut[i % num_elem_persistent];
    }
}

void launch_reset_data(int* streaming_data, int const* persistent_data_lut, size_t const num_elem_streaming, size_t const num_elem_persistent, cudaStream_t stream)
{
    int constexpr threads = 1024;
    int constexpr blocks = 32;

    reset_data<<<blocks, threads, 0, stream>>>(
        streaming_data, persistent_data_lut,
        num_elem_streaming, num_elem_persistent
    );
}

template<typename Fn>
float measure_performance(Fn&& fn, cudaStream_t stream, int const num_warmup = 20, int const num_repeats = 100)
{
    cudaEvent_t start, stop;
    float elapsed_time{};

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    for (int i = 0; i < num_warmup; i++) {
        fn(stream);
    }
    cudaStreamSynchronize(stream);

    // measure performance
    cudaEventRecord(start, stream);
    for (int i = 0; i < num_repeats; i++) {
        fn(stream);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time / num_repeats;
}

int main(int argc, char** argv)
{
    size_t num_mbytes_persistent = 3;
    if (argc > 1) {
        num_mbytes_persistent = std::atoi(argv[1]);
    }

    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << std::left << std::setw(30) << "GPU " << ": " << device_prop.name << "\n"
              << std::setw(30) << "L2 Cache Size " << ": " << device_prop.l2CacheSize / (1024 * 1024) << " MB\n"
              << std::setw(30) << "Max Persistent L2 Cache Size " << ": " << device_prop.persistingL2CacheMaxSize / (1024 * 1024) << " MB\n";
    
    size_t const num_mbytes_streaming = 1024;
    size_t const num_elem_persistent = num_mbytes_persistent * (1024 * 1024) / sizeof(int);
    size_t const num_elem_streaming = num_mbytes_streaming * (1024 * 1024) / sizeof(int);

    std::cout << std::setw(30) << "Persistent Data Size " << ": " << num_mbytes_persistent << " MB\n"
              << std::setw(30) << "Streaming Data Size " << ": " << num_mbytes_streaming << " MB\n\n";

    // data preparation
    std::vector<int> persistent_data_vec_lut(num_elem_persistent, 0);
    for (int i = 0; i < persistent_data_vec_lut.size(); i++) {
        persistent_data_vec_lut[i] = i;
    }
    std::vector<int> streaming_data_vec(num_elem_streaming, 0);

    int* d_persistent_data, *d_streaming_data;
    cudaMalloc(&d_persistent_data, num_elem_persistent * sizeof(int));
    cudaMalloc(&d_streaming_data, num_elem_streaming * sizeof(int));
    cudaMemcpy(d_persistent_data, persistent_data_vec_lut.data(), num_elem_persistent * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // without using persistent L2 cache
    auto func = std::function<void(cudaStream_t)>(std::bind(launch_reset_data, d_streaming_data, d_persistent_data, num_elem_streaming, num_elem_persistent, std::placeholders::_1));
    auto latency = measure_performance(func, stream);
    std::cout << std::setw(60) << "Latency without Persistent L2 Cache " << ": " << latency << " ms\n";

    // using persistent L2 cache (potentially thrashing)
    size_t const num_mbytes_persistent_cache = 3;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, num_mbytes_persistent_cache * (1024 * 1024));
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_persistent_data);
    stream_attr.accessPolicyWindow.num_bytes = num_mbytes_persistent * (1024 * 1024);
    stream_attr.accessPolicyWindow.hitRatio = 1.0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    latency = measure_performance(func, stream);
    std::string title = "Latency with " + std::to_string(num_mbytes_persistent_cache) + "MB Persistent L2 Cache (potentially thrashing)"; 
    std::cout << std::setw(60) << title << ": " << latency << " ms\n";

    // using persistent L2 cache (non-thrashing)
    stream_attr.accessPolicyWindow.hitRatio = std::min(static_cast<double>(num_mbytes_persistent_cache) / num_mbytes_persistent, 1.0);
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    latency = measure_performance(func, stream);
    title = "Latency with " + std::to_string(num_mbytes_persistent_cache) + "MB Persistent L2 Cache (non-thrashing)"; 
    std::cout << std::setw(60) << title << ": " << latency << " ms\n";

    cudaStreamDestroy(stream);
    cudaFree(d_persistent_data);
    cudaFree(d_streaming_data);

    return 0;
}