# Table of Contents

- [Table of Contents](#table-of-contents)
- [Row- and Column-major Order](#row--and-column-major-order)
- [A Layout of WMMA](#a-layout-of-wmma)
- [References](#references)

<br>

# Row- and Column-major Order

Row-major와 column-major order는 행렬과 같은 다차원 배열을 선형 메모리 공간에 저장하는 방식을 의미한다. Row-major order는 요소를 선형 메모리에 저장할 때 하나의 행을 먼저 다 채우고 그 다음 행을 채우는 식의 순서를 의미하고, column-major order는 하나의 열을 먼저 채우고 다음 열을 채우는 식의 순서를 의미한다. 이들을 그림으로 표현하면 다음과 같다.

> Column-major order는 다차원 배열의 요소를 배치하는 방식이며, 실제 메모리 공간은 row-major order의 linear memory 공간이다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Row_and_column_major_order.svg/800px-Row_and_column_major_order.svg.png" height=250px style="display: block; margin: 0 auto; background-color: white"/>

예를 들어, `m x k x n`의 행렬 곱셈에서 행렬 A, B가 row-major인지 column-major인지에 따라서 행렬 A와 B의 차원 순서가 바뀔 수 있는데, 각 경우에 따른 행렬 A, B의 크기는 다음과 같다.

- row-major order A: `m x k`
- column-major order A: `k x m`
- row-major order B: `k x n`
- column-major order B: `n x k`

따라서, 만약 우리가 두 행렬을 곱할 때 row-major인 행렬 A와 row-major인 행렬 B라면 우리가 일반적으로 알고 있는 행렬 곱셈 방식을 사용하면 된다. `C = AB` 행렬 곱셈을 CPU 코드로 작성하면 다음과 같다. 이때, 행렬 C의 메모리 순서는 row-major order라고 가정한다.
```c++
void gemm_row_a_row_b(float const* A, float const* B, float* C, int m, int n, int k)
{
    int const lda = k;
    int const ldb = n;
    int const ldc = n;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            // calculate C[mi][ni] = A[mi][ki] * B[ki][ni]
            float accum{0.f};
            for (int ki = 0; ki < k; ki++) {
                accum += A[mi * lda + ki] * B[ki * ldb + ni];
            }
            C[mi * ldc + ni] = accum;
        }
    }
}
```
위 코드에서 `C[mi][ni]`를 계산할 때, 행렬 A와 행렬 B의 메모리에 접근할 때의 순서를 주의깊게 살펴보자. 가장 내부의 루프에서 `ki`가 1씩 증가하므로, 행렬 A에 접근할 때는 매 반복마다 바로 인접한 다음 요소에 접근하게 된다. 반면 행렬 B의 경우에는 매 반복마다 `ldb`만큼 떨어진 요소에 접근하게 된다. 어느 한 메모리 주소를 읽을 때, 오직 그 메모리만 읽는 것이 아니라 그 주변의 메모리까지 함께 읽는다. 즉, 주변 메모리를 한 번에 캐시에 올려서 읽게 된다. 따라서, 인접한 메모리 주소를 읽는 행렬 A의 경우에는 cache hit로 인해 빠른 로드가 가능하다. 반면, 행렬 B는 `ldb`만큼 떨어진 요소를 읽게 되므로 행렬 A보다는 cache hit가 덜 발생하게 되어 행렬 A를 읽는 것보다는 느리다.

행렬 B의 메모리에 액세스할 때도 cache hit를 달성시키려면 행렬 B를 column-major order로 배치하면 된다. 즉, `k x n`이 었던 행렬 B를 전치시켜 `n x k` 모양으로 요소를 배치시키는 것이다. 예를 들어, `3 x 3` row-major order 행렬을 column-major order 행렬로 바꾸면 다음과 같이 바뀐다.
```
1 2 3    1 4 7
4 5 6 -> 2 5 8
7 8 9    3 6 9
```
그럼 이제 행렬 A와 행렬 B을 곱해서 행렬 C를 계산할 때, 하나의 요소를 구하기 위해서 행렬 A의 한 행과 행렬 B의 한 열을 곱하는 게 아닌, 행렬 A의 한 행과 행렬 B의 한 행을 곱해서 구해야 한다. 이제 row-major order 행렬 A와 column-major order 행렬 B 간의 곱셈은 다음과 같이 구현될 수 있다.
```c++
void gemm_row_a_col_b(float const* A, float const* B, float* C, int m, int n, int k)
{
    int const lda = k;
    int const ldb = k;
    int const ldc = n;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            // calculate C[mi][ni] = A[mi][ki] * B[ni][ki]
            float accum{0.f};
            for (int ki = 0; ki < k; ki++) {
                accum += A[mi * lda + ki] * B[ni * ldb + ki];
            }
            C[mi * ldc + ni] = accum;
        }
    }
}
```
이제 행렬 B 또한 가장 내부의 루프에서 매 반복마다 인접한 다음 요소에 접근하게 되어 높은 cache hit를 달성할 수 있게 된다.

> Row-major order의 행렬 A와 Column-major order의 행렬 B를 곱할 때, 가장 안쪽 루프에서 행렬 A의 row를 읽고 행렬 B의 column을 읽게 된다. 행렬 A의 row에서 각 요소들은 인접한 메모리에 위치하며, 행렬 B 또한 column에서의 각 요소들이 인접한 메모리에 위치한다. 따라서 두 행렬 모두 **cache-friendly**하여 메모리 로드 속도가 빠르다.

이외에도 행렬 A는 column-major, 행렬 B는 row-major인 경우와 행렬 A는 column-major, 행렬 B도 column-major인 경우가 있다. 각 경우에 대한 코드 구현은 각각 다음과 같다.
```c++
void gemm_col_a_row_b(float const* A, float const* B, float* C, int m, int n, int k)
{
    int const lda = m;
    int const ldb = n;
    int const ldc = n;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            // calculate C[mi][ni] = A[ki][mi] * B[ki][ni]
            float accum{0.f};
            for (int ki = 0; ki < k; ki++) {
                accum += A[ki * lda + mi] * B[ki * ldb + ni];
            }
            C[mi * ldc + ni] = accum;
        }
    }
}

void gemm_col_a_col_b(float const* A, float const* B, float* C, int m, int n, int k)
{
    int const lda = m;
    int const ldb = k;
    int const ldc = n;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            // calculate C[mi][ni] = A[ki][mi] * B[ni][ki]
            float accum{0.f};
            for (int ki = 0; ki < k; ki++) {
                accum += A[ki * lda + mi] * B[ni * ldb + ki];
            }
            C[mi * ldc + ni] = accum;
        }
    }
}
```

실제로 위의 구현들의 속도는 다음과 같이 측정되었다.
```
Row-major A(1024x1024) x Row-major B(1024x1024): 1.291 sec
Row-major A(1024x1024) x Col-major B(1024x1024): 0.879 sec
Col-major A(1024x1024) x Row-major B(1024x1024): 1.454 sec
Col-major A(1024x1024) x Col-major B(1024x1024): 1.263 sec
```

행렬 A는 row-major order, 행렬 B는 column-major order일 때의 속도가 가장 빠르게 측정된다.

> 위의 결과는 원하는 행렬 연산이 `C=AB`이기 때문이다. 만약, `C=AB^T`와 같은 변형식을 구한다면 다른 layout 조합에서 성능이 더 좋을 것이다. 모든 경우에 대한 결과를 [Row-Major vs Column-Major](https://leimao.github.io/blog/Row-Major-VS-Column-Major/)에서 잘 셜명해주고 있다.

# A Layout of WMMA

WMMA에서 `matrix_a`와 `matrix_b`의 fragment를 생성할 때, 해당 행렬의 메모리가 row-major(`nvcuda::wmma::row-major`)인지 column-major(`nvcuda::wmma::col-major`)인지를 설정할 수 있다. 이때, CPU에서의 연산과 마찬가지로 GPU 메모리 상에서도 행렬 A는 row-major, 행렬 B는 column-major가 될 때, 최상의 성능을 얻을 수 있을 것이라고 추정할 수 있다.

GPU 메모리 상에서도 각 layout에 따라 4가지 경우가 존재한다. 이번에는 WMMA에서 각 layout 조합에 따라 성능이 어떻게 측정되는지 살펴보자.

이전에 [Overview of Tensor Cores](/cuda/study/21_overview_of_tensor_cores.md)에서 행렬 A와 행렬 B가 row-major order인 경우에서의 성능만 측정해보았는데, 이 코드를 조금 변형하여 사용하였다. 전체 코드는 아래 링크에서 확인할 수 있다.

- [wmma_layout.cu](/cuda/code/wmma_layout/wmma_layout.cu)

우선 커널을 호출하기 위한 함수이다.
```c++
template<typename T1, typename T2>
void mma_wmma(T1 const* A, T1 const* B, T2* C, uint32_t const m, uint32_t const n, uint32_t const k, bool transA, bool transB, cudaStream_t stream = nullptr)
{
    uint32_t lda = !transA ? k : m;
    uint32_t ldb = !transB ? n : k;
    uint32_t ldc = n;
    float const alpha{1.f};
    float const beta{0.f};

    // shape restriction
    int const WMMA_M{16};
    int const WMMA_N{16};
    int const WMMA_K{16};

    int const warp_size{32};
    int const warp_x{4};
    int const warp_y{4};
    dim3 grid_dim, block_dim;
    // 1. each warp processes 16x16 output tile matrix
    // 2. a block processes 64x64 output tile matrix (it means there are `warp_x` x `warp_y` warps = (4x4 warps))
    // 3. consecutive threads are grouped in a warp => blockDim.x must be a multiple of warp_size(32)
    // => block_dim: (128, 4)
    // => grid_dim: (16, 16)
    block_dim.x = warp_x * warp_size;
    block_dim.y = warp_y;
    grid_dim.x = (m + (WMMA_M * warp_x - 1)) / (WMMA_M * warp_x);
    grid_dim.y = (n + (WMMA_N * warp_y - 1)) / (WMMA_N * warp_y);
    
    if (!transA && !transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::row_major,
                        wmma::row_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else if (!transA && transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::row_major,
                        wmma::col_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else if (transA && !transB) {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::col_major,
                        wmma::row_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
    else {
        mma_wmma_kernel<T1,
                        T2,
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        wmma::col_major,
                        wmma::col_major>
            <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, transA, transB, alpha, beta);
    }
}
```
`1024x1024x1024` 행렬 연산을 수행하기 때문에 사실 layout에 따라 달라지는 것이 별로 없으나, 일단 정석대로 구현하였으며, 코드 상에서의 기본적인 내용은 BLAS의 GEMM 연산을 참조하면 도움이 될 수 있다.

다음은 실제 wmma 연산을 수행할 커널 함수이다.
```c++
template<typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
    typename WMMA_A_FLAG_LAYOUT, typename WMMA_B_FLAG_LAYOUT>
__global__
void mma_wmma_kernel(
    T1 const* A, T1 const* B, T2* C, uint32_t const m, uint32_t const n, uint32_t const k,
    uint32_t const lda, uint32_t const ldb, uint32_t const ldc, bool transA, bool transB, float alpha, float beta)
{
    // determine warp index in a 2D grid (128 x 4) 
    // => it means each block has 4x4 warp and processes 64 x 64 matrix
    uint32_t const warp_row_idx = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    uint32_t const warp_col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    // declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_A_FLAG_LAYOUT> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_B_FLAG_LAYOUT> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> frag_acc;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> frag_c;

    // initialize the output
    wmma::fill_fragment(frag_acc, static_cast<T2>(0));

    // perfor the matrix multiplication
    for (uint32_t ki = 0; ki < k; ki += WMMA_K) {
        uint32_t const a_row_idx = !transA ? warp_row_idx * WMMA_M : ki;
        uint32_t const a_col_idx = !transA ? ki : warp_row_idx * WMMA_M;
        uint32_t const b_row_idx = !transB ? ki : warp_col_idx * WMMA_N;
        uint32_t const b_col_idx = !transB ? warp_col_idx * WMMA_N : ki;

        // check bound
        if (a_row_idx < (!transA ? k : m) && a_col_idx < (!transA ? m : k)
            && b_row_idx < (!transB ? k : n) && b_col_idx < (!transB ? n : k)) {
            T1 const* matrix_a_mptr = A + a_row_idx * lda + a_col_idx;
            T1 const* matrix_b_mptr = B + b_row_idx * ldb + b_col_idx;

            // load the inputs
            wmma::load_matrix_sync(frag_a, matrix_a_mptr, lda);
            wmma::load_matrix_sync(frag_b, matrix_b_mptr, ldb);

            // perform the matrix multiplication
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }
    }

    // scale the output by beta, and add it the result scaled by alpha
    uint32_t const c_row_idx = warp_row_idx * WMMA_M;
    uint32_t const c_col_idx = warp_col_idx * WMMA_N;
    if (c_row_idx < m && c_col_idx < n) {
        T2* matrix_c_mptr = C + c_row_idx * ldc + c_col_idx;
        wmma::load_matrix_sync(frag_c, matrix_c_mptr, ldc, wmma::mem_row_major);

        for (uint32_t i = 0; i < frag_c.num_elements; i++) {
            frag_c.x[i] = alpha * frag_acc.x[i] + beta * frag_c.x[i];
        }
        // store the output
        wmma::store_matrix_sync(matrix_c_mptr, frag_c, ldc, wmma::mem_row_major);
    }
}
```

이렇게 구현한 코드로 컴파일 후, 실행한 결과는 다음과 같다.
```
Maxtirx Sizes
- M: 1024
- N: 1024
- K: 1024
(A   * B  ) WMMA Kernel Latency : 0.177 ms (err: 0.003318) / 12104.076 GFlop/s
(A   * B^T) WMMA Kernel Latency : 0.167 ms (err: 0.003323) / 12838.396 GFlop/s
(A^T * B  ) WMMA Kernel Latency : 0.188 ms (err: 0.003317) / 11449.211 GFlop/s
(A^T * B^T) WMMA Kernel Latency : 0.177 ms (err: 0.003316) / 12113.864 GFlop/s
```

CPU와 동일하게 A가 row-major, B가 column-major일 때 가장 빠르고 A가 column-major, B가 row-major일 때 가장 느리게 측정되는 것을 확인할 수 있다.

<br>

# References

- [Wikipedia: Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
- [Programming Tensor Cores: Native Volta Tensor Cores with CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
- [NVIDIA CUDA Documentation: Warp Level Matrix Multiply-Accumulate Instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)
- [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [Row-Major vs Column-Major](https://leimao.github.io/blog/Row-Major-VS-Column-Major/)