#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <string>
#include <functional>
#include <iomanip>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**
 * Baseline
 * This kernel uses atomic operations to accumulate the individual inputs
 * in a single, device-wide visible variable.
 */
// a GPU-visible variable in global memory
__device__ float global_result;
__global__
void reduceAtomicGlobal(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(&global_result, in[idx]);
    }
}

/**
 * Improvement 1
 * Shared memory is much faster than global memory. Each block can accumulate 
 * partial results in isolated block-wide visible memory.
 * It relieves the contention on a single global variable that all therads want
 * access to.
 */
__global__
void reduceAtomicShared(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    // a shared variable for each block
    __shared__ float x;

    // only one thread in each block should initialize this shared variable
    if (threadIdx.x == 0) {
        x = 0.f;
    }

    __syncthreads(); // before reducing, all threads can see the initialization by thread 0

    // every thread in the block adds its input to the shared variable of the block
    if (idx < N) {
        atomicAdd(&x, in[idx]);
    }

    // wait until all threads have done their part
    __syncthreads();

    // once they are all done, only one thread must add the block's partial result
    // to the global variable
    if (threadIdx.x == 0) {
        atomicAdd(&global_result, x);
    }
}

/**
 * Improvement 2 : using a more suitable algorithm
 * We use a iterative algorithm to utilize the GPU that is massively parallel.
 * In each iteration, threads accumulate partial results from the previous iteration.
 * Before, the contented accesses to one location forced the GPU to perform updates
 * sequentially O(N).
 * Now, each thread can access its own, exclusive shared variable in each iteration
 * in parallel, giving an effiective runtime that is closer to O(logN).
 */
template<unsigned int BLOCK_SIZE>
__global__
void reduceShared(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = (idx < N) ? in[idx] : 0.f;

    // log N iterations to complete
    // in each step, a thread accumulates two partial values to form the input
    // for the next iteration
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        // in each iteration, must make sure that all threads are done writing
        // the updates of the previous iteration and the initialization
        __syncthreads();
        if (threadIdx.x < i) {
            data[threadIdx.x] += data[threadIdx.x + i];
        }
    }

    // thread 0 is the last thread to combine two partial results, and writes to
    // global memory, therefore no synchronization is required after the last iteration
    if (threadIdx.x == 0) {
        atomicAdd(&global_result, data[0]);
    }
}

/**
 * Improvement 3: using warp-level primitives to accelerate the final step
 * Warps have a fast lane for communication. They are free to exchange values in registers
 * when they are being scheduled for execution.
 * Warps will be formed from consecutive threads in groups of 32.
 */
template<unsigned int BLOCK_SIZE>
__global__
void reduceShuffle(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = (idx < N) ? in[idx] : 0.f;

    // the reduction in shared memory stops at 32 partial results.
    for (int i = blockDim.x / 2; i > 16; i /= 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            data[threadIdx.x] += data[threadIdx.x + i];
        }
    }

    // the last 32 values can be handled with warp-level primitives
    float x = data[threadIdx.x];
    if (threadIdx.x < 32) {
        // This replace the last 5 iterations of the `atomicShared` kernel function.
        // The mask indicates which therads participate in the shuffle.
        // The value indicates which register should be shuffled.
        // The final parameter gives the source thread from which the current one should
        // receive the shuffled value.
        // In each shuffle, at least half of the threads only participate so they can
        // provide useful data from the previous shuffle for lower threads.
        // To keep the code short, we always let all threads participate, because it is 
        // an error to let threads reach a shuffle instruction that they don't participate in.
        // 
        // API signature: 
        // - float __shfl_sync(unsigned mask, float var, int srcLane, int width=warpSize);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0) {
        atomicAdd(&global_result, x);
    }
}


// template<unsigned int BLOCK_SIZE>
// __global__
// void reduceShuffleUnroll(float const* __restrict__ in, int N)
// {
//     int const idx = blockDim.x * blockIdx.x + threadIdx.x;

//     __shared__ float data[BLOCK_SIZE];
//     data[threadIdx.x] = (idx < N) ? in[idx] : 0.f;

//     // the reduction in shared memory stops at 32 partial results.
//     for (int i = blockDim.x / 2; i > 16; i /= 2) {
//         __syncthreads();
//         if (threadIdx.x < i) {
//             data[threadIdx.x] += data[threadIdx.x + i];
//         }
//     }

//     // unrolling warp
//     if (threadIdx.x < 16) {
//         volatile float* vsmem = data;
//         vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
//         vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
//         vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
//         vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
//         vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
//     }

//     if (threadIdx.x == 0) {
//         atomicAdd(&global_result, data[0]);
//     }
// }

/**
 * Improvement 4
 * Each thread fetch two values at the start
 */
template<unsigned int BLOCK_SIZE>
__global__
void reduceShuffle2(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float data[BLOCK_SIZE];
    // combine two values upon load from global memory
    data[threadIdx.x] = (idx < N/2) ? in[idx] : 0;
    data[threadIdx.x] += (idx + N/2 < N) ? in[idx + N/2] : 0;

    for (int i = blockDim.x / 2; i > 16; i /= 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            data[threadIdx.x] += data[threadIdx.x + i];
        }
    }

    float x = data[threadIdx.x];
    if (threadIdx.x < 32) {
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0) {
        atomicAdd(&global_result, x);
    }
}

/**
 * Addiitonal: using coorperative groups
 */
template<unsigned int BLOCK_SIZE>
__global__
void reduceUsingCG(float const* __restrict__ in, int N)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = (idx < N) ? in[idx] : 0.f;

    // the reduction in shared memory stops at 32 partial results.
    for (int i = blockDim.x / 2; i > 32; i /= 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            data[threadIdx.x] += data[threadIdx.x + i];
        }
    }

    // the last 32 values can be handled with coorperative groups
    auto threadblock = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(threadblock);
    if (warp.meta_group_rank() == 0) { // first warp group only
        int warp_lane = warp.thread_rank();
        float x = data[warp_lane] + data[warp_lane + 32];
        x = cg::reduce(warp, x, cg::plus<float>());
        if (warp_lane == 0) atomicAdd(&global_result, x);
    }
}


int main(int argc, char** argv)
{
    constexpr unsigned int BLOCK_SIZE = 256;
    constexpr unsigned int WARM_UP_ITER = 10;
    constexpr unsigned int TIMING_ITER = 20;
    constexpr unsigned int N = 100000000;

    std::cout << "the number of elements: " << N << "\n\n";

    // prepare host data
    std::vector<float> host_values(N, 0.f);
    // generate random numbers to reduce
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(1.f);
        std::for_each(host_values.begin(), host_values.end(), [&gen, &dist](float& val) { val = dist(gen); });
    }

    // prepare device memory
    float* device_values;
    cudaMalloc(&device_values, sizeof(float) * N);
    cudaMemcpy(device_values, host_values.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    // CPU reduction
    std::cout << " === CPU Reduction ===\n";
    {
        float result;
        const auto start = std::chrono::system_clock::now();
        for (int i = 0; i < TIMING_ITER; i++) {
            result = std::accumulate(host_values.cbegin(), host_values.cend(), 0.0f);
        }
        const auto end = std::chrono::system_clock::now();
        const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count() * 1000.f;
        std::cout << std::left << std::setw(20) << "CPU Reduction" << "\t" << elapsed_time / TIMING_ITER << " ms \t(" << result << ")\n";
    }

    // GPU reduction
    std::cout << "\n === GPU Reduction ===\n";
    const std::tuple<std::string, void(*)(float const*, int), unsigned int> cases[]{
        {"Atomic Global", reduceAtomicGlobal, N},
        {"Atomic Shared", reduceAtomicShared, N},
        {"Reduce Shared", reduceShared<BLOCK_SIZE>, N},
        {"Reduce Shuffle", reduceShuffle<BLOCK_SIZE>, N},
        // {"Reduce Shuffle Unroll", reduceShuffleUnroll<BLOCK_SIZE>, N},
        {"Reduce Shuffle2", reduceShuffle2<BLOCK_SIZE>, N/2 + 1},
        {"Reduce CG", reduceUsingCG<BLOCK_SIZE>, N},
    };

    for (const auto& [name, kernel, num_threads] : cases) {
        // compute grid and block size
        const unsigned int block = BLOCK_SIZE;
        const unsigned int grid = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // warm-up
        for (int i = 0; i < WARM_UP_ITER; i++) {
            kernel<<<grid, block>>>(device_values, N);
        }

        cudaDeviceSynchronize();
        const auto start = std::chrono::system_clock::now();

        float result = 0.f;
        // run kernel
        for (int i = 0; i < TIMING_ITER; i++) {
            cudaMemcpyToSymbol(global_result, &result, sizeof(float));
            kernel<<<grid, block>>>(device_values, N);
        }

        // cudaMemcpyFromSymbol will implicitly synchronize CPU and GPU
        cudaMemcpyFromSymbol(&result, global_result, sizeof(float));

        const auto end = std::chrono::system_clock::now();
        const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count() * 1000.f;
        std::cout << std::left << std::setw(20) << name << "\t" << elapsed_time / TIMING_ITER << " ms \t(" << result << ")\n";
    }

    cudaFree(device_values);

    return 0;
}