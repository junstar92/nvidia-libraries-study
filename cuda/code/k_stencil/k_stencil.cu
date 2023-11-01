/*
- compile command: nvcc -O2 -o one_stencil one_stencil.cu
*/
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_ERROR_CHECK(err) cuda_error_check((err), #err, __FILE__, __LINE__)

inline void cuda_error_check(cudaError_t err, char const* const func, char const* const file, int const num_line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error(" <<  func << "): " << cudaGetErrorString(err) << " at " << file << ":" << num_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__
void one_stencil(int const* A, int* B, int size)
{
    extern __shared__ int smem[];

    // thread id in the block
    int local_id = threadIdx.x;

    // the first index of output element computed by this block
    int block_start_idx = blockIdx.x * blockDim.x;

    // the id of thread in the scope of the grid
    int global_id = block_start_idx + local_id;

    if (global_id >= size) return;

    // fetching into shared memory
    smem[local_id] = A[global_id];
    if (local_id < 2 && blockDim.x + global_id < size) {
        smem[blockDim.x + local_id] = A[blockDim.x + global_id];
    }

    __syncthreads(); // sync before reading from shared memory

    // each thread computes a single output
    if (global_id < size - 2) {
        B[global_id] = (smem[local_id] + smem[local_id + 1] + smem[local_id + 2]) / 3;
    }
}

__global__
void one_stencil_with_rc(int const* A, int* B, int size)
{
    // declare local register cache
    int rc[2];

    // thread id in the warp
    int local_id = threadIdx.x % warpSize;

    // the first index of output element computed by this warp
    int warp_start_idx = blockIdx.x * blockDim.x + warpSize * (threadIdx.x / warpSize);

    // the id of thread in the scope of the grid
    int global_id = local_id + warp_start_idx;

    if (global_id >= size) return;

    // fetch into register cache
    rc[0] = A[global_id];
    if (local_id < 2 && warpSize + global_id < size) {
        rc[1] = A[warpSize + global_id];
    }

    // each thread computes a single output
    int ac = 0;
    int to_share = rc[0];

    for (int i = 0; i < 3; i++) {
        // threads decide what value will be published in the following access
        if (local_id < i) to_share = rc[1];

        // accessing register cache
        unsigned mask = __activemask();
        ac += __shfl_sync(mask, to_share, (local_id + i) % warpSize);
    }

    if (global_id < size - 2) B[global_id] = ac / 3;
}

__global__
void k_stencil(int const* A, int* B, int k, int size)
{
    extern __shared__ int smem[];

    // thread id in the block
    int local_id = threadIdx.x;

    // the first index of output element computed by this block
    int block_start_idx = blockIdx.x * blockDim.x;

    // the id of thread in the scope of the grid
    int global_id = block_start_idx + local_id;

    if (global_id >= size) return;

    // fetching into shared memory
    smem[local_id] = A[global_id];
    if (local_id < 2*k && blockDim.x + global_id < size) {
        smem[blockDim.x + local_id] = A[blockDim.x + global_id];
    }

    __syncthreads(); // sync before reading from shared memory

    // each thread computes a single output
    if (global_id < size - k) {
        int sum = 0;
        for (int i = 0; i < 2*k + 1; i++) {
            sum += smem[local_id + i];
        }
        B[global_id] = sum / (2*k + 1);
    }
}

__global__
void k_stencil_with_rc(int const* A, int* B, int k, int size)
{
    int b_size = size - k;

    // declare local register cache
    int rc[2];

    // thread id in the warp
    int local_id = threadIdx.x % warpSize;

    // the first index of output element computed by this warp
    int warp_start_idx = blockIdx.x * blockDim.x + warpSize * (threadIdx.x / warpSize);

    // the id of thread in the scope of the grid
    int global_id = local_id + warp_start_idx;

    if (global_id >= size) return;

    // fetch into register cache
    rc[0] = A[global_id];
    if (local_id < 2 * k && warpSize + global_id < size) {
        rc[1] = A[warpSize + global_id];
    }

    bool warp_has_inactive_threads = size - warp_start_idx < warpSize;
    int inactive_threads_in_warp = warp_has_inactive_threads ? warp_start_idx + warpSize - size : 0;

    unsigned mask = (0xffffffff) >> (inactive_threads_in_warp);

    // each thread computes a single output
    int ac = 0;
    int to_share = rc[0];

    for (int i = 0; i < 2*k + 1; i++) {
        // Threads decide what value will be published in the following access.
        ac += __shfl_sync(mask, to_share, (local_id + i) & (warpSize - 1));
        to_share += (i == local_id) * (rc[1] - rc[0]);
    }

    if (global_id < b_size) B[global_id] = ac / (2*k + 1);
}

void one_stencil_host(std::vector<int> const& h_A, std::vector<int>& h_B, int n_output_per_thread = 1)
{
    int *d_A, *d_B;

    CUDA_ERROR_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(int)));

    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    float ms = 0.f;
    float total_ms = 0.f;
    int const n_iter = 100;

    dim3 block{1024};
    dim3 grid{(h_A.size() + block.x - 1) / block.x};
    size_t smem_size = (block.x + 2) * sizeof(int);

    // warp-up
    for (int i = 0; i < 10; i++) {
        one_stencil<<<grid, block, smem_size>>>(d_A, d_B, h_A.size());
    }

    // implementation using shared memory
    for (int i = 0; i < n_iter; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        one_stencil<<<grid, block, smem_size>>>(d_A, d_B, h_A.size());
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));

        total_ms += ms;
    }
    printf("one_stencil<<<%d, %d, %ld>>>: %.3f ms\n", grid.x, block.x, smem_size, total_ms / n_iter);
    CUDA_ERROR_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
        printf("%d ", h_B[i]);
    }
    printf("\n");

    // implementation using register cache
    total_ms = 0.f;
    for (int i = 0; i < n_iter; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        one_stencil_with_rc<<<grid, block>>>(d_A, d_B, h_A.size());
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));

        total_ms += ms;
    }
    printf("one_stencil_with_rc<<<%d, %d>>>: %.3f ms\n", grid.x, block.x, total_ms / n_iter);
    CUDA_ERROR_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
        printf("%d ", h_B[i]);
    }
    printf("\n");


    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
}

void k_stencil_host(std::vector<int> const& h_A, std::vector<int>& h_B, int k, int n_output_per_thread = 1)
{
    int *d_A, *d_B;

    CUDA_ERROR_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(int)));

    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    float ms = 0.f;
    float total_ms = 0.f;
    int const n_iter = 100;

    dim3 block{1024};
    dim3 grid{(h_A.size() + block.x - 1) / block.x};
    size_t smem_size = (block.x + k) * sizeof(int);

    // warp-up
    for (int i = 0; i < 10; i++) {
        k_stencil<<<grid, block, smem_size>>>(d_A, d_B, k, h_A.size());
    }

    // implementation using shared memory
    for (int i = 0; i < n_iter; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        k_stencil<<<grid, block, smem_size>>>(d_A, d_B, k, h_A.size());
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));

        total_ms += ms;
    }
    printf("k_stencil<<<%d, %d, %ld>>>: %.3f ms\n", grid.x, block.x, smem_size, total_ms / n_iter);
    CUDA_ERROR_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
        printf("%d ", h_B[i]);
    }
    printf("\n");

    // implementation using register cache
    total_ms = 0.f;
    for (int i = 0; i < n_iter; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        k_stencil_with_rc<<<grid, block>>>(d_A, d_B, k, h_A.size());
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));

        total_ms += ms;
    }
    printf("k_stencil_with_rc<<<%d, %d>>>: %.3f ms\n", grid.x, block.x, total_ms / n_iter);
    CUDA_ERROR_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; i++) {
        printf("%d ", h_B[i]);
    }
    printf("\n");


    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    CUDA_ERROR_CHECK(cudaFree(d_A));
    CUDA_ERROR_CHECK(cudaFree(d_B));
}

int main(int argc, char** argv)
{
    int k = 1;
    int size = 1 << 27;

    if (argc > 1) {
        k = std::stoi(argv[1]);
    }
    if (argc > 2) {
        size = std::stoi(argv[1]);
    }

    printf("> %d-Stencil - size: %d\n", k, size);

    std::vector<int> h_A(size, 0), h_B(size - 2*k, 0);
    for (int i = 0; i < size; i++) {
        h_A[i] = i % 17;
    }

    if (k == 1) {
        one_stencil_host(h_A, h_B);
    }
    else {
        k_stencil_host(h_A, h_B, k);
    }

    return 0;
}