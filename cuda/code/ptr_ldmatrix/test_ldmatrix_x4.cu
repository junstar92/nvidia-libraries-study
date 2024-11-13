#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__
void ldmatrix_test_kernel(half* dst, half const* src, int const count)
{
    __shared__ half shmem[256];

    int tid = threadIdx.x;
    int group_idx = tid / 8;

    if (8 * tid < count) {
        shmem[8 * tid] = src[8 * tid];
        shmem[8 * tid + 1] = src[8 * tid + 1];
        shmem[8 * tid + 2] = src[8 * tid + 2];
        shmem[8 * tid + 3] = src[8 * tid + 3];
        shmem[8 * tid + 4] = src[8 * tid + 4];
        shmem[8 * tid + 5] = src[8 * tid + 5];
        shmem[8 * tid + 6] = src[8 * tid + 6];
        shmem[8 * tid + 7] = src[8 * tid + 7];
    }
    __syncthreads();

    half* shmem_ptr = nullptr;
    half frag[8];
    if (tid < 32) {
        shmem_ptr = shmem + 16 * ((group_idx / 2) * 8 + tid % 8) + (group_idx % 2) * 8;
    }

    uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(frag);
    uint32_t shmem_ptr_val = __cvta_generic_to_shared(shmem_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
        : "r"(shmem_ptr_val)
    );

    // printf("[%2d] %f %f %f %f %f %f %f %f\n", tid,
    //     __half2float(frag[0]), __half2float(frag[1]), __half2float(frag[2]), __half2float(frag[3]),
    //     __half2float(frag[4]), __half2float(frag[5]), __half2float(frag[6]), __half2float(frag[7]));

    if (8 * tid < count) {
        dst[8 * tid] = frag[0];
        dst[8 * tid + 1] = frag[1];
        dst[8 * tid + 2] = frag[2];
        dst[8 * tid + 3] = frag[3];
        dst[8 * tid + 4] = frag[4];
        dst[8 * tid + 5] = frag[5];
        dst[8 * tid + 6] = frag[6];
        dst[8 * tid + 7] = frag[7];
    }
}

int main()
{
    size_t count = 256;
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