# Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview of Tensor Cores](#overview-of-tensor-cores)
- [Warp-synchronous Matix Multiply-Accumulate (WMMA)](#warp-synchronous-matix-multiply-accumulate-wmma)
  - [Simple WMMA Example](#simple-wmma-example)
- [WMMA Instruction (PTX)](#wmma-instruction-ptx)
- [References](#references)

<br>

# Overview of Tensor Cores

NVIDIA 텐서 코어(Tensor Cores)는 mixed precision에서의 GEMM 연산을 가속하는 가속기이다. Volta 아키텍처에서 도입되었고 현재(2023.09) 4세대까지 발전했다. AI에서 핵심이 되는 부분 중 하나가 행렬 연산이므로 이러한 가속기의 역할이 매우 중요하다고 볼 수 있다.

하나의 텐서 코어는 `D = A * B + C` 연산을 수행하는데, 이때, `A`, `B`, `C`, `D`는 아래 그림과 같이 모두 4x4 행렬이다. 텐서 코어는 mixed precision에서의 행렬 연산을 지원하며, 일반적으로 말하는 mixed precision에서 곱셈의 피연산자인 `A`와 `B`는 FP16 행렬이고, 누적 연산에 해당하는 `C`와 `D`는 FP32 행렬이다. INT 연산인 경우에는 `A`와 `B`는 `INT8`(unsigned or signed char)이고, `C`와 `D`는 `INT32`(int)가 될 수도 있다. 이번 포스팅에서는 `FP16`, `FP32`의 mixed precision에 대해서 주로 언급한다.

<img src="https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2017/05/image4.png" height=200px style="display: block; margin: 0 auto; background-color: white"/>

WMMA API를 통해서 텐서 코어를 사용하도록 프로그래밍이 가능한데, WMMA API에서 타입 및 행렬 크기에 대한 제약 사항을 살펴보면 4x4 행렬 연산이 아닌 16x16 행렬 연산을 수행하도록 추상화되어 있는 것처럼 보인다 (요소 타입에 따라 다를 수 있음).

> CUDA C++에서는 오직 warp-level primitive만 지원한다.

실제로 어떻게 처리되는지 이해하려면 하드웨어 수준에서의 동작을 살펴봐야 한다. SM을 자세히 살펴보면 아래와 같이 4개의 sub-core (4 processing blocks)로 나누어져 있다. 4개의 sub-core에는 각각 warp scheduler를 갖는다. 이 구조는 Volta부터 지금까지 유지되고 있는 것으로 보인다. Volta/Turing 아키텍처에서는 하나의 SM에 8개의 텐서 코어가 있고, 각 sub-core들이 각각 2개의 텐서 코어를 사용하게 된다. Ampere 이상의 아키텍처에서는 SM 당 4개의 텐서 코어가 있다.

> 이미지 자료가 Volta/Turing에 대해서만 있어서 우선 Volta/Turning 기준으로 살펴봄

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5Ld6C%2FbtsrgnIcLPq%2F752I8b3LaAxEieXg0kNNgk%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color: white"/>

각 sub-core는 아래와 같은 구조를 갖는다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdIujxc%2FbtsrgRbc4Si%2FlaMCyXuVEmpMRtVjNFQiuk%2Fimg.png" height=600px style="display: block; margin: 0 auto; background-color: white"/>

일반적으로 말하는 `A*B+C` 텐서 코어 연산은 16개의 FP16 요소로 구성된 `A`, 16개의 FP16 요소로 구성된 `B`, 그리고 accumulator에 대한 8개의 FP16 or FP32 요소의 `C`(and `D`)의 fragments들로 수행된다.

<img src="https://images.anandtech.com/doci/12673/HC29.21.132-Volta-Choquette-NVIDIA-Final3-22.png" height=400px style="display: block; margin: 0 auto; background-color: white"/>

그리고, 이들 fragments와 WMMA API(`wmma::mma_sync()`)를 통해서 아래의 행렬 연산을 수행하게 된다.

<img src="https://images.anandtech.com/doci/12673/s8278-cuda-new-features-and-beyond-13.png" height=400px style="display: block; margin: 0 auto; background-color: white"/>

# Warp-synchronous Matix Multiply-Accumulate (WMMA)

[Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)에서 WMMA instruction에서의 텐서 코어 연산 과정을 자세히 분석하고 있다. 이 논문에서는 한 warp 내 32개의 스레드가 모두 동작하는 패턴에서 sub-core가 행렬 곱셈을 어떻게 계산하는지 살펴보고 있다. 개념적으로는 텐서 코어는 4x4 submatrices에서 동작하여 16x16 행렬을 계산한다.

WMMA 행렬 연산을 그림으로 표현하면 다음과 같다. 그림에서 하나의 warp가 16x16 행렬 연산을 담당한다고 보면 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbGsMUw%2FbtsrgRCiMqo%2FUc7RoKgQX1FyKmvKxC0tTK%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color: white"/>

하나의 warp만 사용하여 `D = A * B + C`를 WMMA API를 사용하여 구현하면 다음과 같은 코드 구조를 갖게 된다. 각 행렬의 크기는 아래와 같다.

- A: 16x16 (half)
- B: 16x16 (half)
- C: 16x16 (float)
- D: 16x16 (float)

> 아래 코드에서는 모든 행렬의 메모리가 row-major layout이라고 가정한다.

```c++
using namespace nvcuda;
__device__
void wmma_16x16x16(half* a, half* b, float* c, float* d)
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

    // perfor the matrix multiplication
    wmma::mma_sync(Cmat, Amat, Bmat, Cmat);

    // store the output
    wmma::store_matrix_sync(d, Cmat, 16, wmma::mem_row_major);
}
```

하나의 warp 내에는 32개의 스레드들이 있고, 이 스레드들이 서로 협력하여 행렬 D를 계산하게 된다. 텐서 코어는 4x4x4 행렬 연산에 특화되어 있는데, 실제 fragment의 값을 출력하여 스레드들이 어떤 요소의 계산을 담당하는지 알아볼 수 있다.

> [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf) 논문에서 16x16 행렬 A, B, C의 각 요소들이 어떤 스레드 인덱스로 매핑되는지 잘 보여주고 있다 (A: figure 4.2, B: figure 4.3, C: figure 4.7). 이 논문에서는 column-major layout 기준으로 설명하고 있는데, 위 코드에서는 row-major layout을 사용하고 있다. 따라서, 논문에서의 스레드 인덱스 매핑이 전치된 것이 위 코드의 경우와 일치한다.

논문에서 보여주는 것과 같이 각 요소가 어떤 스레드 인덱스에 매핑되는지 살펴보자.

아래의 커널 함수를
```c++
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
```
연속된 메모리(row-major layout)의 값을 0부터 255까지 채운 16x16 행렬 A,B로 호출하면 아래의 출력을 확인할 수 있다.
```
A host: 
     0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15 
    16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31 
    32     33     34     35     36     37     38     39     40     41     42     43     44     45     46     47 
    48     49     50     51     52     53     54     55     56     57     58     59     60     61     62     63 
    64     65     66     67     68     69     70     71     72     73     74     75     76     77     78     79 
    80     81     82     83     84     85     86     87     88     89     90     91     92     93     94     95 
    96     97     98     99    100    101    102    103    104    105    106    107    108    109    110    111 
   112    113    114    115    116    117    118    119    120    121    122    123    124    125    126    127 
   128    129    130    131    132    133    134    135    136    137    138    139    140    141    142    143 
   144    145    146    147    148    149    150    151    152    153    154    155    156    157    158    159 
   160    161    162    163    164    165    166    167    168    169    170    171    172    173    174    175 
   176    177    178    179    180    181    182    183    184    185    186    187    188    189    190    191 
   192    193    194    195    196    197    198    199    200    201    202    203    204    205    206    207 
   208    209    210    211    212    213    214    215    216    217    218    219    220    221    222    223 
   224    225    226    227    228    229    230    231    232    233    234    235    236    237    238    239 
   240    241    242    243    244    245    246    247    248    249    250    251    252    253    254    255 

B host: 
     0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15 
...
   240    241    242    243    244    245    246    247    248    249    250    251    252    253    254    255 

Thread [0] 
Fragment A:   0   1 128 129   8   9 136 137   0   1 128 129   8   9 136 137 
Fragment B:   0  16 128 144   8  24 136 152   0  16 128 144   8  24 136 152 
Thread [1] 
Fragment A:   2   3 130 131  10  11 138 139   2   3 130 131  10  11 138 139 
Fragment B:  32  48 160 176  40  56 168 184  32  48 160 176  40  56 168 184 
Thread [2] 
Fragment A:   4   5 132 133  12  13 140 141   4   5 132 133  12  13 140 141 
Fragment B:  64  80 192 208  72  88 200 216  64  80 192 208  72  88 200 216 
Thread [3] 
Fragment A:   6   7 134 135  14  15 142 143   6   7 134 135  14  15 142 143 
Fragment B:  96 112 224 240 104 120 232 248  96 112 224 240 104 120 232 248 
...
Thread [31] 
Fragment A: 118 119 246 247 126 127 254 255 118 119 246 247 126 127 254 255 
Fragment B: 103 119 231 247 111 127 239 255 103 119 231 247 111 127 239 255
...
```

위 출력을 위한 전체 코드는 아래에서 확인할 수 있다.

- [wmma_index_check.cu](/cuda/code/simple_wmma/wmma_index_check.cu)

위의 출력을 살펴보면 warp 내 스레드가 어떤 행렬 요소 값을 레지스터로 가져오는지 확인할 수 있다. 예를 들어, warp 내 인덱스가 0인 스레드는 행렬 A의 0, 1, 8, 9, 128, 129, 136, 137번째 요소 값과 행렬 B의 0, 16, 8, 24, 128, 144, 136, 152번째 요소 값을 레지스터로 저장한다는 것을 알 수 있다. 스레드 0,1,2,3에서 담당하는 행렬 A의 fragment와 행렬 B의 fragment가 저장하고 있는 값을 가만히 살펴보면 스레드 0,1,2,3에서의 A fragment는 행렬 A의 첫 번째 행이고, B fragment는 행렬 B의 첫 번째 열이라는 것을 확인할 수 있다. 즉, 16x16 행렬 D에서 (0,0) 좌표의 요소 값을 계산할 때, 0,1,2,3 스레드가 서로 협력하여 계산할 것이라고 생각할 수 있다.

행렬 D의 Fragment에 대해서 각 스레드에서의 출력을 첨부하지는 않았지만, 나머지 출력을 살펴보면 스레드 하나가 최종 행렬에서 8개의 요소를 담당한다는 것을 알 수 있다. 즉, 각 스레드들이 최종 계산된 값 중 8개씩 맡아서 메모리에 저장한다. 스레드 0이 담당하는 행렬 D의 요소의 좌표는 `(0,0), (0,1), (16,0), (16,1), (0,16), (0,17), (16,16), (16,17)` 이다.

> [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf) 논문에서 언급하고 있는 행렬 A, B, C 요소들과 스레드 인덱스와의 매칭이 실제 출력해본 결과와 다른 것으로 보인다. 논문에서 사용한 CUDA 버전(9.0)이 너무 낮아서 지금과는 조금 다를 수도 있을 것 같다. 실제로 PTX 어셈블리를 분석해보면 논문에서 보여주는 명령어와는 조금 다른 모양이다.

## Simple WMMA Example

wmma을 사용하여 행렬 곱셈을 수행하는 커널의 구현은 다음과 같다.
```c++
template<typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
    typename WMMA_A_FLAG_LAYOUT, typename WMMA_B_FLAG_LAYOUT>
__global__
void mma_wmma_kernel(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    // Assume the matrices A, B, and C are row-major layout
    uint32_t const lda = k;
    uint32_t const ldb = n;
    uint32_t const ldc = n;
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
        uint32_t const a_row_idx = warp_row_idx * WMMA_M;
        uint32_t const a_col_idx = ki;
        uint32_t const b_row_idx = ki;
        uint32_t const b_col_idx = warp_col_idx * WMMA_N;

        // check bound
        if (a_row_idx < m && a_col_idx < k && b_row_idx < k && b_col_idx < n) {
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
구현의 기본 컨셉은 위에서 설명한 것과 동일하다. 하나의 warp가 16x16 행렬 곱셈을 담당하며, 이 커널을 호출하는 측에서는 하나의 블록이 4x4 warp로 구성되도록 총 512개의 스레드를 갖도록 해준다. 그리드 차원은 이 블록의 차원과 행렬의 크기에 따라 결정된다. `mma_wmma_kernel` 커널 함수를 호출하는 측은 다음과 같이 구현될 수 있다.
```c++
template<typename T1, typename T2>
void mma_wmma(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k, cudaStream_t stream = nullptr)
{
    float const alpha{1.f};
    float const beta{0.f};

    // shape restriction
    int const WMMA_M{16};
    int const WMMA_N{16};
    int const WMMA_K{16};

    int const warp_size{32};
    int const warp_x{1};
    int const warp_y{1};
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
    
    mma_wmma_kernel<T1,
                    T2,
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    wmma::row_major,
                    wmma::row_major>
        <<<grid_dim, block_dim, 0, stream>>>(A, B, C, m, n, k, alpha, beta);
}
```

전체 코드는 아래에서 확인할 수 있다.

- [simple_wmma.cu](/cuda/code/simple_wmma/simple_wmma.cu)

위 코드는 cpu matmul, cuda kernel matmul, wmma matmul을 각각 100번씩 실행하여 평균 실행 시간을 측정한다. 출력 결과는 다음과 같다.
```
Matrix Size
- M: 1024
- N: 1024
- K: 1024
(f32f32f32) CUDA Kernel Latency : 1.282 ms (err: 0.000007) / 1675.189 GFlop/s
(f16f16f32) CUDA Kernel Latency : 1.241 ms (err: 0.002839) / 1729.852 GFlop/s
(f16f16f32) WMMA Kernel Latency : 0.174 ms (err: 0.003318) / 12309.397 GFlop/s
```
꽤나 인상적인 결과를 보여주고 있다. 단순 쿠다 코어만 사용했을 때 1024 x 1024 x 1024 행렬 곱셈은 약 1.2 ms가 걸린다. 하지만, 텐서 코어를 사용하면 10배 정도 빠른 약 0.1 ms만 걸린다는 것을 확인할 수 있다.

지금까지는 하나의 warp가 16x16 행렬 계산을 수행하는 경우에 대해서 살펴봤는데, CUDA 버전이 올라가면서 아래와 같은 모양의 행렬 곱셈도 가능하게 되었다.

<img src="https://images.anandtech.com/doci/12673/s8278-cuda-new-features-and-beyond-16.png" height=400px style="display: block; margin: 0 auto; background-color: white"/>

# WMMA Instruction (PTX)

[Warp Matrix Functions](/cuda/doc/01_programming_guide/07-24_warp_matrix_functions.md)에서 확인할 수 있듯이 CUDA C++ API인 WMMA를 통해서 Tensor Core를 사용하여 행렬 연산을 수행할 수 있다. 하지만 PTX 명령어를 살펴보면 `wmma`뿐만 아니라 `mma` 명령어로 존재한다. 초기에는 `wmma` 명령어(PTX ISA Ver 6.0)만 있었으나 이후에 `mma` 명령어(PTX ISA Ver 6.4)도 추가된 것으로 보인다. Warpgroup Level MMA인 `wgmma` 명령어도 있다.

`wmma`과 `mma` 명령어는 비슷하지만 조금 더 세부적으로 컨트롤할 수 있는 여부의 차이가 있는 것 같다. `wmma`의 경우에는 `wmma.load`와 `wmma.store` 등의 명령어를 제공하여 정해진 제약 조건에 맞는 fragment를 로드하거나 저장한다. 하지만 `mma`의 경우에는 다양한 크기의 matrix fragment를 지원한다. 그리고 `ldmatrix`, `stmatrix` 등의 명령어로 명시적인 fragment 로드 및 저장이 필요한 것으로 확인된다.

> 결론적으로 `mma`를 사용하면 처리할 수 있는 행렬 크기가 조금 더 다양해질 수 있다.

> `wmma`을 사용하면 `half` 타입인 경우, 16x16x16(or 32x8x16, 8x32x16) 사이즈의 행렬 연산만 가능하다. 하지만 `mma` 명령어를 사용하면 다양한 크기의 fragment를 사용할 수 있다. CUDA 문서 내 [Warp Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)에서 자세히 설명하고 있다. 
> 
> 문서 내에서는 특정 fragment에서 각 행렬 요소 값들이 어떤 스레드에 매핑되는지 자세히 보여준다. `wmma`에서 각 요소가 스레드에 매핑되는 것과 유사하게 매핑되는 것으로 보인다. 특히 [Figure 79](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-ldmatrix-fragments)에서는 8x8 크기의 행렬 fragment에서 각 요소가 어떤 스레드에 매핑되는지 보여준다. `wmma`에서는 16x16 행렬에서 각 요소들이 매핑되는 것을 살펴보면 4개의 8x8 행렬을 따로 구분헀을 때, 각 8x8 행렬에서 각 스레드가 담당하는 요소의 위치가 동일한 것을 볼 수 있다. 그리고, 이때 8x8 행렬에서 각 스레드가 담당하는 요소는 [Figure 79](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-ldmatrix-fragments)에서 8x8 행렬에서 각 스레드에 매핑되는 요소와 정확히 일치하는 것을 볼 수 있다.

단, 일반적인 CUDA C/C++로는 구현할 수 없으며 PTX instruction을 직접 사용하여 구현해야 한다는 단점이 있다. 하지만, `wmma`를 사용했을 때 연산할 수 있는 행렬 크기에 제약이 있지만, PTX의 `mma` 명령어를 직접 사용하여 커널을 구현하게 되면 이러한 제약을 조금이나마 피해서 다양한 크기의 행렬 연산을 구현할 수 있을 것으로 보인다.

<br>

# References

- [Warp Matrix Functions](/cuda/doc/01_programming_guide/07-24_warp_matrix_functions.md)
- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [A Shallow Dive Into Tensor Cores](https://www.anandtech.com/show/12673/titan-v-deep-learning-deep-dive/3)
- [CUTLASS: CUDA Template Library for Dense Linear Algebra at All Levels and Scales](https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)
- [Programming Tensor Cores: Native Volta Tensor Cores with CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
- [RTX ON - THE NVIDIA TURING GPU](https://old.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf)
- [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)
- [NVIDIA CUDA Documentation: Warp Level Matrix Multiply-Accumulate Instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Tips for Optimizing GPU Performance Using Tensor Cores](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)
- [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)