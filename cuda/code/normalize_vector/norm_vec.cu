/*
- compile command: nvcc -O2 -arch=sm70 -o norm_vec norm_vec.cu
*/
#include <iostream>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;

#define CUDA_ERROR_CHECK(err) cuda_error_check((err), #err, __FILE__, __LINE__)

inline void cuda_error_check(cudaError_t err, char const* const func, char const* const file, int const num_line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error(" <<  func << "): " << cudaGetErrorString(err) << " at " << file << ":" << num_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template<bool WriteSquareRoot>
__device__
void reduce_block_data(
    cuda::barrier<cuda::thread_scope_block>& barrier,
    cg::thread_block_tile<32>& tile32, double& thread_sum, double* result
)
{
    extern __shared__ double tmp[];

    #pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        thread_sum += tile32.shfl_down(thread_sum, offset);
    }
    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = thread_sum;
    }

    auto token = barrier.arrive();
    barrier.wait(std::move(token));

    // The warp 0 will perform last round of reduction
    if (tile32.meta_group_rank() == 0) {
        double beta = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.;

        #pragma unroll
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            beta += tile32.shfl_down(beta, offset);
        }

        if (tile32.thread_rank() == 0) {
            if (WriteSquareRoot)
                *result = sqrt(beta);
            else
                *result = beta;
        }
    }
}

__global__
void norm_vec_by_dot_product(float* vecA, float* vecB, double* partialResults, int size)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    cg::sync(cta);

    double thread_sum = 0.;
    for (int i = grid.thread_rank(); i < size; i+= grid.size()) {
        thread_sum += (double)(vecA[i] * vecB[i]);
    }

    // Each thread block performs reduction of partial dot products and 
    // writes to global memory
    reduce_block_data<false>(barrier, tile32, thread_sum, &partialResults[blockIdx.x]);

    cg::sync(grid);

    // One block performs the final summation of partial dot products
    // of all the thread blocks and writes the sqrt of final dot product
    if (blockIdx.x == 0) {
        thread_sum = 0.;
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
            thread_sum += partialResults[i];
        }
        reduce_block_data<true>(barrier, tile32, thread_sum, &partialResults[0]);
    }
    cg::sync(grid);

    const double final_value = partialResults[0];

    // Perform normalization of vector A & B
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        vecA[i] = (float)vecA[i] / final_value;
        vecB[i] = (float)vecB[i] / final_value;
    }
}


int main(int argc, char** argv)
{
    int device_id = 0;
    CUDA_ERROR_CHECK(cudaSetDevice(device_id));

    int size = 10000000;
    float *h_vecA, *h_vecB;
    float *d_vecA, *d_vecB;
    float *d_partial_results;

    std::cout << argv[0] << " starting... - size: " << size << std::endl;

    CUDA_ERROR_CHECK(cudaMallocHost(&h_vecA, sizeof(float) * size));
    CUDA_ERROR_CHECK(cudaMallocHost(&h_vecB, sizeof(float) * size));
    CUDA_ERROR_CHECK(cudaMalloc(&d_vecA, sizeof(float) * size));
    CUDA_ERROR_CHECK(cudaMalloc(&d_vecB, sizeof(float) * size));

    float base_val = 2.f;
    for (int i = 0; i < size; i++) {
        h_vecA[i] = h_vecB[i] = base_val;
    }

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    // Data copy
    CUDA_ERROR_CHECK(cudaMemcpyAsync(d_vecA, h_vecA, sizeof(float) * size, cudaMemcpyHostToDevice, stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(d_vecB, h_vecB, sizeof(float) * size, cudaMemcpyHostToDevice, stream));

    // Kernel configuration
    // one-dimensional grid and one-dimensional blocks are configured.
    int min_grid_size = 0, block_size = 0;
    CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, norm_vec_by_dot_product, 0, size));
    
    int smem_size = ((block_size / 32) + 1) * sizeof(float);

    int num_block_per_sm = 0;
    CUDA_ERROR_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_block_per_sm, norm_vec_by_dot_product, block_size, smem_size
    ));

    int multi_processor_count = 0;
    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id));

    min_grid_size = multi_processor_count * num_block_per_sm;
    CUDA_ERROR_CHECK(cudaMalloc(&d_partial_results, sizeof(float) * min_grid_size));

    std::cout << "Launching norm_vec_by_dot_product kernel with num_block = "
        << min_grid_size << " / block_size = " << block_size << " / smem_size = " << smem_size << "bytes" << std::endl;
    dim3 grid{min_grid_size}, block{block_size};
    
    void *kernel_args[] = {&d_vecA, &d_vecB, &d_partial_results, &size};
    CUDA_ERROR_CHECK(cudaLaunchCooperativeKernel((void*)norm_vec_by_dot_product, grid, block, kernel_args, smem_size, stream));

    CUDA_ERROR_CHECK(cudaMemcpyAsync(h_vecA, d_vecA, sizeof(float) * size, cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    float expected_results = (base_val / sqrt(size * base_val * base_val));
    unsigned int matches = 0;
    for (int i = 0; i < size; i++) {
        if (std::abs(h_vecA[i] - expected_results) > 0.00001) {
            std::cout << "mismatch at i = " << i << "(" << h_vecA[i] << " != " << expected_results << ")" << std::endl;
            break;
        }
        else {
            matches++;
        }
    }

    std::cout << "Result = " << ((matches == size) ? "PASSED\n" : "FAILED\n");

    CUDA_ERROR_CHECK(cudaFree(d_partial_results));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
    CUDA_ERROR_CHECK(cudaFree(d_vecA));
    CUDA_ERROR_CHECK(cudaFree(d_vecB));
    CUDA_ERROR_CHECK(cudaFreeHost(h_vecA));
    CUDA_ERROR_CHECK(cudaFreeHost(h_vecB));
}