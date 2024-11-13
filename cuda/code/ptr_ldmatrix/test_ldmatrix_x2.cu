#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__
void ldmatrix_test_kernel(half* dst, half const* src, int const count)
{
    __shared__ half shmem[128];

    int tid = threadIdx.x;

    if (4 * tid < count) {
        shmem[4 * tid] = src[4 * tid];
        shmem[4 * tid + 1] = src[4 * tid + 1];
        shmem[4 * tid + 2] = src[4 * tid + 2];
        shmem[4 * tid + 3] = src[4 * tid + 3];
    }
    __syncthreads();

    half* shmem_ptr = nullptr;
    half frag[4];
    if (tid < 16) {
        shmem_ptr = shmem + 16 * (tid % 8) + (tid / 8) * 8;
    }

    uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(frag);
    uint32_t shmem_ptr_val = __cvta_generic_to_shared(shmem_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 {%0, %1}, [%2];"
        : "=r"(reg_ptr[0]), "=r"(reg_ptr[1])
        : "r"(shmem_ptr_val)
    );

    // printf("[%2d] %f %f %f %f\n", tid, __half2float(frag[0]), __half2float(frag[1]), __half2float(frag[2]), __half2float(frag[3]));

    if (4 * tid < count) {
        dst[4 * tid] = frag[0];
        dst[4 * tid + 1] = frag[1];
        dst[4 * tid + 2] = frag[2];
        dst[4 * tid + 3] = frag[3];
    }
}

int main()
{
    size_t count = 128;
    std::vector<half> h_src, h_dst;

    h_src.resize(count);
    h_dst.resize(count);
    for (size_t i = 0; i < count; i++) {
        h_src[i] = half(static_cast<float>(i));
    }
    std::cout << "Source Matrix:\n";
    for (size_t i = 0; i < count; i++) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(1) << static_cast<float>(h_src[i]) << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << "\n";
        }
    }

    half *d_src_f16, *d_dst_f16;
    cudaMalloc(&d_src_f16, count * sizeof(half));
    cudaMalloc(&d_dst_f16, count * sizeof(half));

    cudaMemcpy(d_src_f16, h_src.data(), count * sizeof(half), cudaMemcpyHostToDevice);

    ldmatrix_test_kernel<<<1, 32>>>(d_dst_f16, d_src_f16, count);

    cudaMemcpy(h_dst.data(), d_dst_f16, count * sizeof(half), cudaMemcpyDeviceToHost);

    std::cout << "\nResult Matrix:\n";
    for (size_t i = 0; i < count; i++) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(1) << float(h_dst[i]) << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << "\n";
        }
    }

    cudaFree(d_src_f16);
    cudaFree(d_dst_f16);
}