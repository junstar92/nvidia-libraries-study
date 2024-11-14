#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__
void mma_m16n8k16_fp16(
    float* c,
    half const* a, half const* b
)
{
    uint32_t const *A = reinterpret_cast<uint32_t const*>(a);
    uint32_t const *B = reinterpret_cast<uint32_t const*>(b);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3])
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__device__
void mma_m16n8k16_fp16(
    half* c,
    half const* a, half const* b
)
{
    uint32_t const *A = reinterpret_cast<uint32_t const*>(a);
    uint32_t const *B = reinterpret_cast<uint32_t const*>(b);
    uint32_t* C = reinterpret_cast<uint32_t*>(c);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1},"
        "{%2, %3, %4, %5},"
        "{%6, %7},"
        "{%8, %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3])
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1])
    );
}

__global__
void mma_m16n8k16_fp16_kernel(half const* A, half const* B, half* C, int const M, int const N, int const K)
{
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int group_id = lane_id >> 2;
    int tid_in_group = lane_id % 4;

    half a_frag[8], b_frag[4];
    float c_frag[4]{};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        auto const* tmp_A = A + (i % 2) * (M / 2) * K + (i / 2) * (K / 2) + group_id * K + tid_in_group * 2;
        auto* tmp_a_frag = a_frag + i * 2;
        *reinterpret_cast<float*>(tmp_a_frag) = *reinterpret_cast<float const*>(tmp_A);
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto const* tmp_B = B + (i % 2) * (K / 2) + group_id * K + tid_in_group * 2;
        auto* tmp_b_frag = b_frag + i * 2;
        *reinterpret_cast<float*>(tmp_b_frag) = *reinterpret_cast<float const*>(tmp_B);
    }

    mma_m16n8k16_fp16(c_frag, a_frag, b_frag);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto* tmp_C = C + (i % 2) * (M / 2) * N + group_id * N + tid_in_group * 2;
        auto* tmp_c_frag = c_frag + i * 2;
        half2 val = __float22half2_rn(*reinterpret_cast<float2*>(tmp_c_frag));
        *reinterpret_cast<float*>(tmp_C) = *reinterpret_cast<float*>(&val);
    }
}

__device__
void ldm4(half* dst, half const* src)
{
    uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(dst);
    uint32_t shmem_ptr = __cvta_generic_to_shared(src);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
        : "r"(shmem_ptr)
    );
}

__device__
void ldm2(half* dst, half const* src)
{
    uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(dst);
    uint32_t shmem_ptr = __cvta_generic_to_shared(src);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg_ptr[0]), "=r"(reg_ptr[1])
        : "r"(shmem_ptr)
    );
}

__global__
void mma_m16n8k16_fp16_kernel2(half const* A, half const* B, half* C, int const M, int const N, int const K)
{
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int group_id = lane_id >> 2;
    int tid_in_group = lane_id % 4;

    __shared__ half a_shmem[16*16], b_shmem[8*16];
    *reinterpret_cast<float4*>(a_shmem + 8 * tid) = *reinterpret_cast<float4 const*>(A + 8 * tid);
    *reinterpret_cast<float2*>(b_shmem + 4 * tid) = *reinterpret_cast<float2 const*>(B + 4 * tid);

    half a_frag[8], b_frag[4], c_frag[4]{};

    half* a_shmem_row = a_shmem + (lane_id % 16) * K + ((lane_id >> 3) / 2) * (K / 2);
    half* b_shmem_row = b_shmem + (lane_id % 8) * K + ((lane_id >> 3) % 2) * (K / 2);

    ldm4(a_frag, a_shmem_row);
    ldm2(b_frag, b_shmem_row);

    mma_m16n8k16_fp16(c_frag, a_frag, b_frag);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto* tmp_C = C + (i % 2) * (M / 2) * N + group_id * N + tid_in_group * 2;
        auto* tmp_c_frag = c_frag + i * 2;
        *reinterpret_cast<float*>(tmp_C) = *reinterpret_cast<float*>(tmp_c_frag);
    }
}

__global__
void mma_fp16_ref_kernel(half const* A, half const* B, half* C, int const M, int const N, int const K)
{
    int const col = blockDim.x * blockIdx.x + threadIdx.x;
    int const row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < N) {
        half acc = 0.f;
        for (int i = 0; i < K; i++) {
            acc = __hadd(acc, __hmul(A[row * K + i], B[col * K + i]));
        }
        C[row * N + col] = acc;
    }
}

int main()
{
    int const m{16}, n{8}, k{16};

    std::vector<half> A, B, C, C_ref;
    A.resize(m*k);                  // row-major
    B.resize(k*n);                  // column-major
    C.resize(m*n, half(0.f));       // row-major
    C_ref.resize(m*n, half(0.f));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i * k + j] = half(float(i * k + j) / 1e2f);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B[i * k + j] = half(float(i * k + j) / 1e2f);
        }
    }


    half *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, m * k * sizeof(half));
    cudaMalloc(&d_B, n * k * sizeof(half));
    cudaMalloc(&d_C, m * n * sizeof(half));
    cudaMalloc(&d_C_ref, m * n * sizeof(half));

    cudaMemcpy(d_A, A.data(), m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(half), cudaMemcpyHostToDevice);

    // ref result
    mma_fp16_ref_kernel<<<1, {16,16}>>>(d_A, d_B, d_C_ref, m, n, k);
    cudaMemcpy(C_ref.data(), d_C_ref, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "Ref Results:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(4) << float(C_ref[i * n + j]) << " ";
        }
        std::cout << "\n";
    }

    // using mma inst
    mma_m16n8k16_fp16_kernel<<<1, 32>>>(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C.data(), d_C, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "\nDevice Results:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(4) << float(C[i * n + j]) << " ";
        }
        std::cout << "\n";
    }

    // using mma with ldmatrix
    C.clear(); C.resize(m * n, half(0.f));
    mma_m16n8k16_fp16_kernel2<<<1, 32>>>(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C.data(), d_C, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "\nDevice Results with ldmatrix:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(4) << float(C[i * n + j]) << " ";
        }
        std::cout << "\n";
    }
}