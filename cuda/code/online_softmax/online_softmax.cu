// nvcc -o online_softmax online_softmax.cu -std=c++17 -O3
#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <iomanip>
#include <cfloat>

template<typename T>
struct MD
{
    T m;
    T d;
};

template<typename T>
struct SumOp
{
    __host__ __device__ __forceinline__
    T operator()(T a, T b) {
        return a + b;
    }

    __host__ __device__ __forceinline__
    T init_value() const {
        return {};
    }
};

template<typename T>
struct MaxOp
{
    __host__ __device__ __forceinline__
    T operator()(T a, T b) {
        return std::max(a, b);
    }

    __host__ __device__ __forceinline__
    T init_value() const {
        return {};
    }
};

template<>
struct MaxOp<float>
{
    __host__ __device__ __forceinline__
    float operator()(float a, float b) {
#ifndef __CUDACC__
        return std::max(a, b);
#else
        return max(a, b);
#endif
    }

    __host__ __device__ __forceinline__
    float init_value() const {
        return FLT_MIN;
    }
};

template<typename T>
struct MDOp
{
    __host__ __device__ __forceinline__
    MD<T> operator()(MD<T> a, MD<T> b) {
        bool is_bigger = a.m > b.m;
        auto bigger = is_bigger ? a : b;
        auto smaller = is_bigger ? b : a;
        
        MD<T> ret;
        ret.m = bigger.m;
        ret.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
        return ret;
    }

    __host__ __device__ __forceinline__
    MD<T> init_value() const {
        MD<T> ret;
        ret.m = MaxOp<T>().init_value();
        ret.d = SumOp<T>().init_value();
        return ret;
    }
};

template<typename T, typename Op>
__device__ __forceinline__
T warp_reduce(T const val, Op op)
{
    T ret = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        ret = op(ret, __shfl_xor_sync(0xffffffff, ret, mask));
    }

    return ret;
}

template<typename T, typename Op>
__device__ __forceinline__
MD<T> warp_reduce(MD<T> const val, Op op)
{
    MD<T> ret = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MD<T> new_md;
        new_md.d = __shfl_xor_sync(0xffffffff, ret.d, mask);
        new_md.m = __shfl_xor_sync(0xffffffff, ret.m, mask);
        ret = op(ret, new_md);
    }

    return ret;
}


template<typename T, typename Op>
__device__ __forceinline__
T block_reduce(T const val, Op op)
{
    __shared__ T shared[32];
    int lane_id = threadIdx.x & 0x1f;
    int warp_id = threadIdx.x >> 5;

    T block_partial = warp_reduce(val, op);

    if (lane_id == 0) {
        shared[warp_id] = block_partial;
    }
    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    block_partial = is_mask ? shared[lane_id] : op.init_value();

    return warp_reduce(block_partial, op);
}

__global__
void native_softmax(float const* __restrict__ x, float* __restrict__ y, int const V)
{
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    x += batch_idx * V;
    y += batch_idx * V;

    __shared__ float d_total_inverse;
    SumOp<float> sum_op;

    float d_partial = 0.f;
    for (int i = thread_idx; i < V; i += blockDim.x) {
        d_partial += __expf(x[i]);
    }
    float d_total = block_reduce(d_partial, sum_op);
    if (thread_idx == 0) {
        d_total_inverse = __fdividef(1.f, d_total);
    }
    __syncthreads();

    for (int i = thread_idx; i < V; i += blockDim.x) {
        y[i] = __expf(x[i]) * d_total_inverse;
    }
}

__global__
void safe_softmax(float const* __restrict__ x, float* __restrict__ y, int const V)
{
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    x += batch_idx * V;
    y += batch_idx * V;

    __shared__ float m_shared;
    __shared__ float d_total_inverse;
    SumOp<float> sum_op;
    MaxOp<float> max_op;

    float m_partial = FLT_MIN;
    for (int i = thread_idx; i < V; i += blockDim.x) {
        m_partial = max_op(m_partial, x[i]);
    }
    float m_total = block_reduce(m_partial, max_op);
    if (thread_idx == 0) {
        m_shared = m_total;
    }
    __syncthreads();

    float d_partial = 0.f;
    for (int i = thread_idx; i < V; i += blockDim.x) {
        d_partial += __expf(x[i] - m_shared);
    }
    float d_total = block_reduce(d_partial, sum_op);
    if (thread_idx == 0) {
        d_total_inverse = __fdividef(1.f, d_total);
    }
    __syncthreads();

    for (int i = thread_idx; i < V; i += blockDim.x) {
        y[i] = __expf(x[i] - m_shared) * d_total_inverse;
    }
}

__global__
void online_softmax(float const* __restrict__ x, float* __restrict__ y, int const V)
{
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    x += batch_idx * V;
    y += batch_idx * V;

    __shared__ MD<float> md_shared;
    MDOp<float> md_op;

    MD<float> md_partial;
    md_partial.m = FLT_MIN;
    md_partial.d = 0.f;
    for (int i = thread_idx; i < V; i += blockDim.x) {
        MD<float> new_md;
        new_md.m = x[i];
        new_md.d = 1.f;
        md_partial = md_op(md_partial, new_md);
    }
    MD<float> md_total = block_reduce(md_partial, md_op);
    if (thread_idx == 0) {
        md_shared = md_total;
    }
    __syncthreads();

    float d_total_inverse = __fdividef(1.f, md_shared.d);
    for (int i = thread_idx; i < V; i += blockDim.x) {
        y[i] = __expf(x[i] - md_shared.m) * d_total_inverse;
    }
}

void native_softmax_host(float const* __restrict__ x, float* __restrict__ y, int const V, int const batch_size)
{
    for (int batch = 0; batch < batch_size; ++batch) {
        float d_total = 0.f;
        for (int i = 0; i < V; ++i) {
            d_total += std::exp(x[i]);
        }
        for (int i = 0; i < V; ++i) {
            y[i] = std::exp(x[i]) / d_total;
        }
        x += V;
        y += V;
    }
}

void init_vector(std::vector<float>& vec)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist;

    for (auto& elem : vec) {
        elem = dist(gen);
    }
}


float run_benchmark(void (*kernel)(float const*, float*, int const), float const* x, float* y, int const V, int const BATCH_SIZE, int const num_warmup = 10, int const num_iterations = 100)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    for (int i = 0; i < num_warmup; ++i) {
        kernel<<<BATCH_SIZE, 256>>>(x, y, V);
    }
    
    // benchmark
    float total_elapsed_time = 0.f;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernel<<<BATCH_SIZE, 256>>>(x, y, V);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_elapsed_time / num_iterations;
}

void compare_softmax_results(float const* y_host, float const* y_device, int const V, int const BATCH_SIZE)
{
    int const num_elements = V * BATCH_SIZE;
    float max_diff = 0.f;
    double total_diff = 0.0;

    for (int i = 0; i < num_elements; ++i) {
        float diff = std::abs(y_host[i] - y_device[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }
    std::cout << "  (max diff: " << max_diff << " / avg diff: " << (float)(total_diff / num_elements) << ")\n";
}

int main(int argc, char** argv)
{
    int V = 4096, BATCH_SIZE = 1000;
    int num_elements = V * BATCH_SIZE;

    std::cout << " - BATCH_SIZE: " << BATCH_SIZE << "\n"
            << " - VECTOR_SIZE: " << V << "\n\n";

    std::vector<float> x(num_elements), y_ref(num_elements, 0.f);
    init_vector(x);

    // native softmax on host side
    native_softmax_host(x.data(), y_ref.data(), V, BATCH_SIZE);

    float *d_x, *d_y;
    cudaMalloc(&d_x, num_elements * sizeof(float));
    cudaMalloc(&d_y, num_elements * sizeof(float));
    cudaMemcpy(d_x, x.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    std::tuple<std::string, void(*)(float const*, float*, int)> cases[]{
        {"native_softmax", &native_softmax},
        {"safe_softmax", &safe_softmax},
        {"online_softmax", &online_softmax},
    };
    for (auto const& [func_name, kernel_func] : cases) {
        auto elapsed_time = run_benchmark(kernel_func, d_x, d_y, V, BATCH_SIZE);
        std::cout << std::setw(20) << func_name << " : " << std::fixed << std::setprecision(6) << elapsed_time << " ms";

        std::vector<float> y(num_elements, 0.f);
        cudaMemcpy(y.data(), d_y, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        compare_softmax_results(y_ref.data(), y.data(), V, BATCH_SIZE);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}