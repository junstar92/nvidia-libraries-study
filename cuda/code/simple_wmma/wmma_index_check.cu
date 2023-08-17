/*
 * Print element index that threads in a warp load in register.
 * 
 * compile command: nvcc -o wmma_index_check wmma_index_check.cu -arch sm_86 (compute capability >= 72)
 */
#include <iostream>
#include <vector>
#include <iomanip>

#include <cuda_runtime.h>
#include <mma.h>

#define M 16
#define N 16
#define K 16

#define CUDA_ERROR_CHECK(err) cuda_error_check((err), #err, __FILE__, __LINE__)

using namespace nvcuda;

void cuda_error_check(cudaError_t err, char const* const func, char const* const file, int const num_line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error(" <<  func << "): " << cudaGetErrorString(err) << " at " << file << ":" << num_line << std::endl;
        std::exit(EXIT_FAILURE);
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
            std::cout << std::setw(6) << std::setprecision(4) << mat[i * col + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__global__
void wmma_index_check(half* a, half* b, float* d)
{
    // declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Amat;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Bmat;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> Cmat;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> Dmat;

    // initialize the output to zero
    wmma::fill_fragment(Cmat, 0.f);

    // load the inputs
    wmma::load_matrix_sync(Amat, a, 16);
    wmma::load_matrix_sync(Bmat, b, 16);
    for (int i = 0; i < warpSize; i++) {
        if (threadIdx.x == i) {
            printf("Thread [%d] \n", threadIdx.x);
            printf("Fragment A: ");
            for (uint32_t i = 0; i < Amat.num_elements; i++) {
                printf("%3.0f ", __half2float(Amat.x[i]));
            }
            printf("\nFragment B: ");
            for (uint32_t i = 0; i < Bmat.num_elements; i++) {
                printf("%3.0f ", __half2float(Bmat.x[i]));
            }
            printf("\n");
        }
    }

    // perfor the matrix multiplication
    wmma::mma_sync(Cmat, Amat, Bmat, Cmat);
    for (int i = 0; i < warpSize; i++) {
        if (threadIdx.x == i) {
            printf("Thread [%d] Fragment C: ", threadIdx.x);
            for (uint32_t i = 0; i < Cmat.num_elements; i++) {
                printf("%3.0f ", Cmat.x[i]);
            }
            printf("\n");
        }
    }

    // store the output
    wmma::store_matrix_sync(d, Cmat, 16, wmma::mem_row_major);
}

int main(int argc, char** argv)
{
    // allocate host memory
    std::vector<float> h_float_matrix_a(M * K);
    std::vector<float> h_float_matrix_b(K * N);
    std::vector<half> h_matrix_a(M * K);
    std::vector<half> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N, 0.f);
    // initialize host memory
    for (size_t i = 0; i < h_float_matrix_a.size(); i++) {
        h_float_matrix_a[i] = i;
        h_float_matrix_b[i] = i;
    }
    float2half(h_matrix_a.data(), h_float_matrix_a.data(), h_float_matrix_a.size());
    float2half(h_matrix_b.data(), h_float_matrix_b.data(), h_float_matrix_b.size());

    // allocate device memory
    half *d_matrix_a{nullptr}, *d_matrix_b{nullptr};
    float *d_matrix_c{nullptr};
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_a, M * K * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_b, K * N * sizeof(half)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_matrix_c, M * N * sizeof(float)));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_a, h_matrix_a.data(), h_matrix_a.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_matrix_b, h_matrix_b.data(), h_matrix_b.size() * sizeof(half), cudaMemcpyHostToDevice));

    print(h_float_matrix_a, M, K, "A host: ");
    print(h_float_matrix_b, K, N, "B host: ");

    // wmma mma
    dim3 block_dim{32};
    dim3 grid_dim{1};
    wmma_index_check<<<grid_dim, block_dim>>>(d_matrix_a, d_matrix_b, d_matrix_c);
    CUDA_ERROR_CHECK(cudaMemcpy(h_matrix_c.data(), d_matrix_c, h_matrix_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    print(h_matrix_c, M, N, "C dev :  ");


    // free resources
    CUDA_ERROR_CHECK(cudaFree(d_matrix_a));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_b));
    CUDA_ERROR_CHECK(cudaFree(d_matrix_c));

    return 0;
}