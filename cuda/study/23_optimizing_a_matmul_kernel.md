# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Kernel 1: Naive Implementation](#kernel-1-naive-implementation)
  - [Lower Bound and Upper Bound](#lower-bound-and-upper-bound)
  - [Memory Bandwidth of Naive Implementation Kernel](#memory-bandwidth-of-naive-implementation-kernel)
- [Kernel 2: Shared Memory Cache-Blocking](#kernel-2-shared-memory-cache-blocking)
  - [Occupancy Calculation](#occupancy-calculation)
- [Kernel 3: 1D Blocktiling for Calculating Multiple Results per Thread](#kernel-3-1d-blocktiling-for-calculating-multiple-results-per-thread)
- [Kernel 4: 2D BLocktiling for Increasing Arithmetic Intensity](#kernel-4-2d-blocktiling-for-increasing-arithmetic-intensity)
- [Kernel 5: Vectorize SMEM and GMEM Accesses](#kernel-5-vectorize-smem-and-gmem-accesses)
- [Kernel 6: Warptiling](#kernel-6-warptiling)
- [Results for Each Size](#results-for-each-size)
- [References](#references)

<br>

# Introduction

CUDA에서의 행렬 곱셈 최적화에 대해 잘 정리한 블로그를 발견했는데, 최적화 과정과 설명이 잘 되어 있어서 따로 정리해보고자 한다. 참고한 블로그는 아래와 같다.

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

행렬 곱셈은 딥러닝에서 거의 대부분의 연산을 차지하기 때문에 딥러닝에서 매우 중요한 연산이라고 할 수 있다. 먼저 가장 기본적인 구현부터 시작하여 최적화를 적용해볼텐데, 최적화의 목표는 cuBLAS의 `SGEMM`과 유사한 성능의 커널을 구현하는 것이다. 사용되는 최적화 기법에는 global memory에 대한 coalesced access pattern, shared memory caching, occupancy optimization 등의 기법이 포함된다.

우선 행렬의 각 차원의 크기가 `M = N = K = 4096`이라고 가정하고 마지막에 다양한 크기에 대한 성능도 비교할 예정이다.

`M = N = K = 4096`인 경우, 각 커널의 성능은 다음과 같다.
|No|Kernel|Average Time|TFLOPs/s|relative to cuBLAS|
|-:|------|-----------:|-------:|-----------------:|
|0|cuBLAS| 6.519 | 19.641 | 100% |
|1|Naive SGEMM| 88.501 | 1.447 | 7.37% |
|2|SMEM SGEMM| 57.474 | 2.228 | 11.34% |
|3|1D BlockTile SGEMM| 19.204 | 6.668 | 33.95% |
|4|2D BlockTile SGEMM| 12.105 | 10.578 | 53.86% |
|5|Vectorize SGEMM| 8.615 | 14.863 | 75.67% |
|6|Warptiling SGEMM| 10.087 | 12.694 | 64.63% |
|-|cuBLAS| 6.985 | 18.331 | 100% |

전체 코드는 [link](/cuda/code/perf_sgemm)에서 확인할 수 있다.

# Kernel 1: Naive Implementation

가장 심플한 구현은 하나의 thread가 output matrix의 하나의 요소를 계산하는 방식으로 구현할 수 있다. 이를 그림으로 표현하면 다음과 같다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/naive-kernel.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

CUDA의 thread 계층에 대해서 알고 있다면 가장 단순하게 구현할 수 있는 방법이다. 이를 CUDA 커널 함수로 구현하면 다음과 같다.
```c++
__global__
void naive_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x; // col
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y; // row

    // bound check
    if (y < m && x < n) {
        float acc = 0.f;
        for (unsigned int i = 0; i < k; i++) {
            acc += A[y * k + i] * B[i * n + x];
        }
        C[y * n + x] = alpha * acc + beta * C[y * n + x];
    }
}
```

## Lower Bound and Upper Bound

해당 커널은 최적화가 하나도 들어가지 않은 가장 기본적인 커널 구현이므로 이 구현의 성능을 lower bound라고 생각할 수 있다. $\alpha \mathit{A} @ \mathit{B} + \beta \mathit{C}$을 연산하게 되는데, 이 연산에서의 total flops, memory read/store는 다음과 같다 (4096 x 4096 행렬 두 개를 곱하고 4096 x 4096 행렬을 더하는 경우).

- Total FLOPS: 2*MNK + 3*MN = $2\times 4096^3 + 3\times4096^2$ FLOPS = 약 128 GFLOPS
- Total Memory Read: (MK + KN + MN) * 4B = $3\times4096^2\times 4\mathit{B}$ = 192 MB
- Total Memory Store: MN * 4B = $4096^2 \times 4\mathit{B}$ = 64 MB

최소 256 MB의 메모리에 대해서 global memory transfer이 필요하며 이는 매우 큰 수치라는 것을 알 수 있다. 이 정보를 이용하여 커널 성능의 상한을 계산할 수 있다. 예를 들어, RTX3080의 경우, FP32 연산의 최대 성능은 30TFLOPs/s, global memory의 최대 bandwidth는 760GB/s라고 알려져 있다. 따라서, 위 수치를 달성하려면, 연산은 약 4.1ms이고 memory transfer는 약 0.33ms 이어야 한다. 수치적으로 연산이 메모리 액세스보다 10배 이상의 시간이 걸린다는 것을 볼 수 있다. 이는 커널이 compute bound라는 것을 의미한다.

## Memory Bandwidth of Naive Implementation Kernel

바로 위에서 구현한 커널을 조금 더 자세히 살펴보자. 동일한 블록에서 스레드의 좌표가 (0, 0)과 (0, 1)인 두 스레드는 A의 동일한 행을 로드하고, B의 서로 다른 열을 로드한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEJotH%2FbtsytZ1OMMB%2FDsdgDroyQOD3p60HFbY340%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

따라서, 행렬 A에서 동일한 행을 읽을 때에는 블록 내 동일한 행 좌표의 스레드들은 모두 A의 동일한 행을 읽으므로 브로드캐스트로 동작한다. 그리고 열에 인접한 좌표의 스레드들은 서로 연속된 좌표의 값을 읽으므로 coalesced access 패턴을 달성하게 된다. 또한, C의 메모리에 저장할 때에도 인접한 스레드들이 인접한 메모리의 요소에 값을 저장하므로 이때도 coalesced access 패턴을 만족한다.

# Kernel 2: Shared Memory Cache-Blocking

GPU는 shared memory (SMEM)이라는 on-chip 메모리를 가지고 있다. 물리적으로 SM 당 하나의 shared memory를 가지며, 이 메모리는 블록 내 분배되는 리소스이다. 스레드들은 이 메모리를 통해 서로 커뮤니케이션을 할 수 있다. RTX 3080의 경우에 shared memory에 대한 제약은 다음과 같다.

- Total amount of shared memory per block: 48152 Bytes (48 KBytes)
- Total shared memory per multiprocessor:  102400 Bytes (100 KBytes)

Shared memory는 on-chip에 위치하기 때문에 global memory에 비해서 latency가 낮고 bandwidth가 높다. 따라서, 이번에는 shared memory를 사용하여 SGEMM 커널을 구현한다. 구현에 대한 자세한 내용은 [Shared Memory](https://github.com/junstar92/nvidia-libraries-study/blob/main/cuda/doc/01_programming_guide/03-02-04_shared_memory.md)에서 자세히 언급하고 있다. Shared memory를 이용한 행렬 곱셈 커널 연산을 그림으로 표현하면 다음과 같다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/cache-blocking.png" height=450px style="display: block; margin: 0 auto; background-color:white"/>

커널 구현은 아래와 같다.
```c++
template<int BLOCK_SIZE>
__global__
void smem_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int c_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    unsigned int c_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    float acc = 0.f;
    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // calculate row, column, data index
        a_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        a_col = BLOCK_SIZE * block + threadIdx.x;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_SIZE * block + threadIdx.y;
        b_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        As[threadIdx.y][threadIdx.x] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    if (c_row < m && c_col < n) {
        C[c_row * n + c_col] = alpha * acc + beta * C[c_row * n + c_col];
    }
}
```

이 커널을 통해 global memory를 사용한 커널 대비 약 30% 이상의 성능 향상을 달성했다 (2.109 TFLOPs/s). 하지만, RTX 3080에서 제공하는 30 TFLOPs/s보다는 현저하게 떨어지는 수치이다.

테스트에서 `BLOCK_SIZE`는 32로 설정했다. 따라서, 하나의 스레드 블록에서 `2*32*32*4B = 8KB` 크기의 shared memory를 사용한다. RTX 3080에서는 각 블록에서 사용할 수 있는 shared memory 크기가 최대 48KB이다. 각 멀티프로세스(SM)에서는 최대 100KB의 shared memory를 사용할 수 있다. 만약, 커널에서 48 KB의 SMEM을 모두 사용하도록 수정한다면, 각 SM에서는 오직 2개의 스레드 블록만 동시에 유지할 수 있다. 즉, 블록 당 SMEM 사용률을 높이면 점유율(occupancy)가 떨어질 수 있다.

> 점유율(occupancy)은 SM 당 active warps의 수와 SM 당 가능한 최대 active warps의 수의 비율로 정의된다.

점유율이 높으면 이용 가능한 issuable instruction 풀이 커지므로 각 연산의 높은 latency를 숨길 수 있어서 매우 유용하다. SM에서 더 많은 active blocks를 로드하는데, 아래의 세 가지 제한 사항이 있다.

- register count
- warp count
- SMEM capacity

## Occupancy Calculation

`M = N = K = 4096` 크기의 예제를 통해 점유율을 계산해보자. 먼저 `cudaGetDeviceProperties()` API를 통해 쿼리한 GPU 장치의 정보는 다음과 같다.
```
Device 0: "NVIDIA GeForce RTX 3080"
  CUDA Driver Version / Runtime Version          11.8 / 11.8
  CUDA Capability Major/Minor version number:    8.6
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Warp size:                                     32
  Maximum number of warps per multiprocessor:    48
  ( 68) Multiprocessors, (128) CUDA Cores/MP:    8704 CUDA Cores
  Total amount of global memory:                 9984 MBytes (10469376000 Bytes)
  Total amount of constant memory:               65536 Bytes (64 KBytes)
  Total amount of shared memory per block:       49152 Bytes (48 KBytes)
  Total shared memory per multiprocessor:        102400 Bytes (100 KBytes)
  Total number of registers available per block: 65536
  Total number of registers available per SM:    65536
```

그리고 `--ptxas-options=-v` 컴파일 옵션을 통해 아래의 커널 정보를 얻을 수 있다.
```
ptxas info    : Compiling entry function '_Z17smem_sgemm_kernelILi32EEviiifPKfS1_fPf' for 'sm_86'
ptxas info    : Function properties for _Z17smem_sgemm_kernelILi32EEviiifPKfS1_fPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 37 registers, 8192 bytes smem, 400 bytes cmem[0]
```

여기서 아래의 정보를 얻을 수 있다. `BLOCK_SIZE`의 크기는 32로 설정했으므로, 블록 당 스레드의 갯수는 1024개라는 것을 알 수 있다.

- Register per Thread: 37
- SMEM per Block     : 8192B

작업은 블록 단위로 SM에 스케쥴링되며, SM은 이를 수용할 수 있는 충분한 리소스가 있으면 더 많은 블록을 로드한다. 각 리소스에 대해 로드할 수 있는 블록의 갯수를 계산하면 다음과 같다.

- Shared Memory: 8192 Bytes/block + 1024 Bytes/block(for CUDA runtime use) = 9216 Bytes/block. 102400 Bytes/SM / 9216 Bytes/block = 11.11 => 11 Blocks
- Threads: 1024 threads per block (max threads per SM: 1536) => 1 Block
- Registers: 37 registers per thread * 32 threads per warp = 1184 registers per warp. 레지스터 할당은 256개의 레지스터 단위로 warp에 할당되므로 하나의 워프에 1280개의 레지스터가 할당된다. 블록 당 32개의 워프가 있으므로 한 블록에서 사용하는 레지스터의 갯수는 1280 * 32 = 40960 registers per block (max registers per SM: 65536) => 1 Block

위 계산을 통해 이 커널은 블록 당 스레드의 수와 블록 당 레지스터의 수에 의해서 로드할 수 있는 블록의 수가 크게 제한된다는 것을 알 수 있다. 하나의 SM에서 하나의 블록만 로드할 수 있으므로 점유율은 `32 active waprs / 48 max active warps = 67%` 이다.

67%의 점유율은 사실 그리 나쁜편은 아니다. 따라서, 점유율로 이 커널의 속도가 느린 이유를 설명할 수 없다. 이는 `Nsight Compute`를 통해 실행된 명령어 통계를 살펴보면 힌트를 얻을 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEDrnS%2FbtsyynVXPok%2FAuejNMCLEZuR7PyPiv49O1%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

`LDS`와 `FFMA`가 큰 비중을 차지하고 있다는 것을 알 수 있다. 이 커널의 [PTX](https://godbolt.org/z/sYhvz4d6b)를 살펴보면 inner loop(dot-product in block)에서 아래의 명령어들이 연속되어서 사용되는 것을 볼 수 있다 (SASS의 `LDS`와 `FFMA`가 아래의 두 명령어에 해당되는 것으로 보인다).
```
ld.shared.f32   %f32, [%r9+768];
ld.shared.f32   %f33, [%r8+24];
fma.rn.f32      %f34, %f33, %f32, %f31;
```

메모리 로드는 단순한 FMA보다 latency가 크고, 구현해야 할 커널이 compute-bound이어야 한다는 점을 고려하면 이러한 패턴은 좋지 않다. `Nsight Compute`로 워프의 상태에 대한 통계를 살펴보면 이러한 패턴의 문제를 확인할 수 있다. 이 통계는 실행된 명령 당 각 상태에서 사용된 사이클 수를 정량화한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fl5f1m%2FbtsyFlQ4aY2%2FokSvsAzIKQO3sp9PKYgkA0%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

`Stall MIO Throttle`이 큰 비중을 차지하고 있다는 것을 알 수 있다. 이 상태에 대한 의미는 다음과 같다.

> Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure.

설명에 따르면 special math instruction, dynamic branches, shared memory를 포함하는 MIO pipline을 극도로 활용하는 경우에 이 상태가 높을 수 있다고 언급하고 있다. 우리가 구현한 커널에서 특별한 수학 명령어나 동적 분기는 없으므로 shared memory에 의해서 지연되었다는 것을 확인할 수 있다.

그렇다면 SMEM instructions를 줄이려면 어떻게 해야 할까?

한 가지 방법은 각 스레드가 두 개 이상의 요소를 계산하도록 하는 것이 있다. 이를 통해 레지스터에서 더 많은 작업들을 수행하고 shared memory에 대한 의존도를 줄일 수 있다.

# Kernel 3: 1D Blocktiling for Calculating Multiple Results per Thread

다음으로 구현할 커널에서는 이전 커널과 유사하지만 각 스레드에서 결과 행렬의 여러 요소들을 계산하기 위해 새로운 inner loop를 추가한다. 이 커널에서는 `BM * BK + BK * BN = 64*8 + 64*8 = 1024 float`, 즉, 블록 당 4KB의 shared memory를 사용한다. 아래 그림에서 두 개의 개별 스레드가 새로운 inner loop에서 액세스하는 값을 주황색과 빨간색으로 보여주고 있다.

> `BM`, `BK`, `BN`은 행렬 차원에 대한 블록 크기에 해당한다. 이전 커널에서는 M, N, K에 대해서 동일한 크기를 적용했었다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/kernel_4_1D_blocktiling.png" height=800px style="display: block; margin: 0 auto; background-color:white"/>

커널 구현은 다음과 같다.

```c++
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M>
__global__
void smem_1d_blocktiling_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_K - 1) / BLOCK_K;
    unsigned int const thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int const a_thread_row = thread_idx / BLOCK_K;
    unsigned int const a_thread_col = thread_idx % BLOCK_K;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx, c_row, c_col;

    float Tacc[THREAD_M] = {0.f, };

    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    #pragma unroll
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_M][BLOCK_K];
        __shared__ float Bs[BLOCK_K][BLOCK_N];
        // calculate row, column, data index of A & B
        a_row = BLOCK_M * blockIdx.y + a_thread_row;
        a_col = BLOCK_K * block + a_thread_col;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_K * block + threadIdx.y;
        b_col = BLOCK_N * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        As[a_thread_row][a_thread_col] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i++) { // dot product loop
            // we make dot product loop, which facilitates
            // resue of the Bs element, which we can cache in a tmp register.
            float Btmp = Bs[i][threadIdx.x];
            #pragma unroll
            for (int t = 0; t < THREAD_M; t++) { // inner loop
                Tacc[t] += As[threadIdx.y + THREAD_M * t][i] * Btmp;
            }
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    #pragma unroll
    for (int t = 0; t < THREAD_M; t++) {
        c_row = BLOCK_M * blockIdx.y + t * THREAD_M + threadIdx.y;
        c_col = BLOCK_N * blockIdx.x + threadIdx.x;
        if (c_row < m && c_col < n) {
            C[c_row * n + c_col] = alpha * Tacc[t] + beta * C[c_row * n + c_col];
        }
    }
}
```

위 커널은 단순히 SMEM만 사용한 이전 커널(`smem_sgemm_kernel`)에 비해 약 3배 정보 빨라졌다는 것을 확인할 수 있다.

먼저 이전 커널 `smem_sgemm_kernel`에서 각 스레드에서 메모리 액세스가 얼마나 수행되는지 계산해보자.

- Global Memory: `K/32` iterations of outer loop * 2 loads
- Shared Memory: `K/32` iterations of outer loop * `BLOCK_SIZE`(=`32`) * 2 loads
- Memory Access Per A Element: `K/16` GMEM + `K*2` SMEM

방금 구현한 `smem_1d_blocktiling_sgemm_kernel`에서의 메모리 액세스는 다음과 같이 계산할 수 있다.

- Global Memory: `K/8` iterations of outer loop * 2 loads
- Shared Memory: `K/8` iterations of outer loop * `BLOCK_K`(=`8`) * (1 + `TM`(=`8`)) loads
- Memory Access Per A Element: `K/32` GMEM + `K*9/8` SMEM

Nsight System으로 `smem_1d_blocktiling_sgemm_kernel` 커널을 프로파일링해보면 이전 커널 대비 `Stall MIO Throttle`의 사이클 수가 크게 감소했다는 것을 확인할 수 있다. 또한, Warp Cycles Per Issued Instruction과 Warp Cycles Per Executed Instruction 또한 36.05에서 19.43으로 감소했다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnwqyV%2FbtsyESn4XWJ%2FK0iQVFuiftSy95EnKrHgr1%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

# Kernel 4: 2D BLocktiling for Increasing Arithmetic Intensity

SMEM만 사용한 두 번째 커널(`smem_sgemm_kernel`)보다 성능은 좋아졌지만, 여전히 cuBLAS 성능과는 아직 차이가 크다. 따라서, 이전 커널 구현에서 적용한 최적화를 다시 한 번 적용하여 스레드에서 더 많은 계산을 수행하도록 한다. 그러면 커널이 더 빠르게 실행될 수 있는데, 이는 **arithmetic intensity**가 증가하기 때문이다. 아래 그림에서는 한 스레드에서 더 많은 연산을 수행하면 arithmetic intensity가 왜 증가하게 되는지를 시각적으로 보여준다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/raising_arith_inten.png" height=700px style="display: block; margin: 0 auto; background-color:white"/>

스레드에서 더 많은 요소를 처리할 때, 결과적으로 FLOPs의 수는 동일하지만 요소 하나당 메모리에 액세스하는 횟수가 줄어들게 된다.

아래 코드는 2D Blocktiling을 적용한 커널 구현이며, 테스트에서 사용한 각 템플릿 인자의 값은 다음과 같다.

- `BLOCK_M`: 64
- `BLOCK_N`: 64
- `BLOCK_K` = `THREAD_M` = `THREAD_N` = 8

```c++
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__
void smem_2d_blocktiling_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int const n_blocks = (k + BLOCK_K - 1) / BLOCK_K;
    unsigned int const thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int const a_thread_row = thread_idx / BLOCK_K;
    unsigned int const a_thread_col = thread_idx % BLOCK_K;
    unsigned int const b_thread_row = thread_idx / (BLOCK_N / THREAD_N);
    unsigned int const b_thread_col = thread_idx % (BLOCK_N / THREAD_N);
    unsigned int row, col, offset;

    float Tacc[THREAD_M][THREAD_N] = {0.f,};
    float At[THREAD_M] = {0.f,};
    float Bt[THREAD_N] = {0.f,};

    // loop over all the sub-matrices of A and B to compute the block sub-matrix
    #pragma unroll
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float As[BLOCK_M][BLOCK_K];
        __shared__ float Bs[BLOCK_K][BLOCK_N];

        // load the matrices from global memory to shared memory
        offset = BLOCK_M / THREAD_M; // row offset for A
        #pragma unroll
        for (unsigned int o = 0; o < BLOCK_M; o += offset) {
            row = BLOCK_M * blockIdx.y + a_thread_row + o;
            col = BLOCK_K * block + a_thread_col;
            As[a_thread_row + o][a_thread_col] = (row < m && col < k) ? A[row * k + col] : 0.f;
        }
        offset = BLOCK_N / THREAD_N; // col offset for B
        #pragma unroll
        for (unsigned int o = 0; o < BLOCK_N; o += offset) {
            row = BLOCK_K * block + b_thread_row;
            col = BLOCK_N * blockIdx.x + b_thread_col + o;
            Bs[b_thread_row][b_thread_col + o] = (row < k && col < n) ? B[row * n + col] : 0.f;
        }
        cta.sync(); // sync to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (unsigned int i = 0; i < BLOCK_K; i++) { // dot product loop
            // block into registers
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_M; t++) {
                At[t] = As[a_thread_row * THREAD_M + t][i];
            }
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_N; t++) {
                Bt[t] = Bs[i][b_thread_col * THREAD_N + t];
            }

            #pragma unroll
            for (unsigned int tm = 0; tm < THREAD_M; tm++) {
                #pragma unroll
                for (unsigned int tn = 0; tn < THREAD_N; tn++) {
                    Tacc[tm][tn] += At[tm] * Bt[tn];
                }
            }
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    #pragma unroll
    for (unsigned int tm = 0; tm < THREAD_M; tm++) {
        #pragma unroll
        for (unsigned int tn = 0; tn < THREAD_N; tn++) {
            row = BLOCK_M * blockIdx.y + THREAD_M * threadIdx.y + tm;
            col = BLOCK_N * blockIdx.x + THREAD_N * threadIdx.x + tn;
            if (row < m && col < n) {
                C[row * n + col] = alpha * Tacc[tm][tn] + beta * C[row * n + col];
            }
        }
    }
}
```

위 커널은 하나의 스레드에서 행렬 C의 `8 * 8`개의 요소를 계산한다. 첫 번째 단계에서 커널은 모든 스레드가 협력하여 global memory로부터 필요한 값들을 SMEM cache를 채운다. 그런 다음 각 스레드는 관련된 SMEM 요소 값을 곱하고 그 결과를 레지스터에 누적한다. 이는 아래 코드에 해당한다.
```c++
float Tacc[THREAD_M][THREAD_N] = {0.f,};
float At[THREAD_M] = {0.f,};
float Bt[THREAD_N] = {0.f,};

// loop over all the sub-matrices of A and B to compute the block sub-matrix
#pragma unroll
for (unsigned int block = 0; block < n_blocks; block++) {
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    // load the matrices from global memory to shared memory
    offset = BLOCK_M / THREAD_M; // row offset for A
    #pragma unroll
    for (unsigned int o = 0; o < BLOCK_M; o += offset) {
        row = BLOCK_M * blockIdx.y + a_thread_row + o;
        col = BLOCK_K * block + a_thread_col;
        As[a_thread_row + o][a_thread_col] = (row < m && col < k) ? A[row * k + col] : 0.f;
    }
    offset = BLOCK_N / THREAD_N; // col offset for B
    #pragma unroll
    for (unsigned int o = 0; o < BLOCK_N; o += offset) {
        row = BLOCK_K * block + b_thread_row;
        col = BLOCK_N * blockIdx.x + b_thread_col + o;
        Bs[b_thread_row][b_thread_col + o] = (row < k && col < n) ? B[row * n + col] : 0.f;
    }
    cta.sync(); // sync to make sure the matrices are loaded

    // multiply the two matrices
    #pragma unroll
    for (unsigned int i = 0; i < BLOCK_K; i++) { // dot product loop
        // block into registers
        #pragma unroll
        for (unsigned int t = 0; t < THREAD_M; t++) {
            At[t] = As[a_thread_row * THREAD_M + t][i];
        }
        #pragma unroll
        for (unsigned int t = 0; t < THREAD_N; t++) {
            Bt[t] = Bs[i][b_thread_col * THREAD_N + t];
        }

        #pragma unroll
        for (unsigned int tm = 0; tm < THREAD_M; tm++) {
            #pragma unroll
            for (unsigned int tn = 0; tn < THREAD_N; tn++) {
                Tacc[tm][tn] += At[tm] * Bt[tn];
            }
        }
    }
    cta.sync();
}
```

Outer loop에서 해당 반복에서 처리할 A, B 요소의 값을 SMEM cache로 복사한다. 그런 다음 inner loop에서 `BLOCK_K` 값만큼 반복하며 dot product를 수행한다. 각 반복에서 핊요한 A, B 요소는 다시 SMEM에서 레지스터로 복사되며, 레지스터에 복사된 값이 마지막 중첩된 inner loop에서 서로 곱해져서 다시 레지스터에 누적된다. Dot product loop는 아래 단계로 진행된다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/kernel_5_reg_blocking.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

1D Blocktiling에 비해 약 1.6배 가량 성능이 향상된 것을 확인할 수 있다. 이 커널에 대한 메모리 액세스를 계산하면 다음과 같다.

- Global Memory: `K/8` outer loop iterations * `2` (A and B) * 8 loads
- Shared Memory: `K/8` outer loop iterations * `8` (dot product loop) * `2` (A and B) * 8 loads
- Memory Access Per A Element: `K/32` GMEM + `K/4` SMEM

참고로 1D Blocktiling 커널의 메모리 액세스는 다음과 같다.

- Memory Access Per A Element: `K/32` GMEM + `K*9/8` SMEM

이 커널에 대한 instruction 통계를 살펴보면 다음과 같이 `LDS`의 수가 이전에 비해 엄청 줄어든 것을 확인할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbk7j2E%2FbtsyEAOwtHA%2FzhEP8k4Dj8hUGaKq9d2KY1%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

# Kernel 5: Vectorize SMEM and GMEM Accesses

성능이 처음에 비해서 많이 좋아졌지만, 이전 커널의 warp 통계를 보다시피 stalled warp가 너무 빈번하게 발생한다. 조금 더 성능을 향상시키기 위해서 이번 커널에서는 아래의 두 가지 방법을 적용한다.

- Transposing As to enable auto-vectorization of SMEM loads
- Promising the compiler alignment on the GMEM accesses

첫 번째 최적화는 `As`를 전치하는 것이다. 이를 통해 `As`를 로드할 때 vectorized SMEM loads (`LDS.128` in SASS)를 사용할 수 있다. 이전 커널 구현에서 3개의 inner loop를 시각화한 것과 동일한데, `As`를 전치한 것만 다르다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/kernel_6_As_transpose.png" height=800px style="display: block; margin: 0 auto; background-color:white"/>

참고 자료를 살펴보면 이를 통해 vectorize 명령어로 로드할 수 있다고 하는데, [link](https://godbolt.org/z/PPnEojf9s)에서 컴파일된 SASS 코드를 보면 `As` 데이터를 레지스터로 로드할 때 이미 `LDS.128`를 사용하는 것으로 보인다 (`#pragma unroll`에 의한 컴파일러 최적화 영향으로 추정된다). 실제로 `As`만 전치한 경우에 별 다른 성능 향상을 발견하지 못했다.

다음으로 적용할 최적화는 GMEM에 대한 load 및 store를 vector datatype인 `float4`를 사용하여 vectorize하는 것이다. 커널 구현에서 인덱싱이 vectorize에 적합하지 않다고 생각되어, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)에서 구현한 방법을 따라서 구현하였다. 해당 페이지에서 구현한 코드에서 global memory to shared memory load에서 잘못된 부분이 있어서 해당 부분을 수정하였다.

먼저 global to shared memory load 코드는 다음과 같다.
```c++
// load the matrices from global memory to shared memory
// transpose A at this point
#pragma unroll
for (unsigned int offset = 0; offset < BLOCK_M; offset += a_stride) {
    float4 tmp = *reinterpret_cast<float4 const*>(&A[(a_inner_row + offset) * k + a_inner_col * 4]);
    As[(a_inner_col * 4 + 0) * BLOCK_M + a_inner_row + offset] = tmp.x;
    As[(a_inner_col * 4 + 1) * BLOCK_M + a_inner_row + offset] = tmp.y;
    As[(a_inner_col * 4 + 2) * BLOCK_M + a_inner_row + offset] = tmp.z;
    As[(a_inner_col * 4 + 3) * BLOCK_M + a_inner_row + offset] = tmp.w;
}
#pragma unroll
for (unsigned int offset = 0; offset < BLOCK_K; offset += b_stride) {
    *reinterpret_cast<float4*>(&Bs[(b_inner_row + offset) * BLOCK_N + b_inner_col * 4]) =
        *reinterpret_cast<float4 const*>(&B[(b_inner_row + offset) * n + b_inner_col * 4]);
}
cta.sync(); // sync to make sure the matrices are loaded
```

전체 커널 구현은 다음과 같다.
```c++
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__
void vectorize_sgemm_kernel(
    int const m, int const n, int const k,
    float const alpha,
    float const* A, float const* B,
    float const beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    unsigned int const num_threads = BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N);
    assert(num_threads == blockDim.x);

    // row, col index in block
    unsigned int const thread_row = threadIdx.x / (BLOCK_N / THREAD_N);
    unsigned int const thread_col = threadIdx.x % (BLOCK_N / THREAD_N);

    // set blocktile to beginning of A's row and B's column
    A += blockIdx.y * BLOCK_M * k;
    B += blockIdx.x * BLOCK_N;
    C += blockIdx.y * BLOCK_M * n + blockIdx.x * BLOCK_N;

    // calculate the indices that this thread will load into SMEM:
    // - load 128bit => 4 elements per thread at each step
    unsigned int const a_inner_row = threadIdx.x / (BLOCK_K / 4);
    unsigned int const a_inner_col = threadIdx.x % (BLOCK_K / 4);
    unsigned int const b_inner_row = threadIdx.x / (BLOCK_N / 4);
    unsigned int const b_inner_col = threadIdx.x % (BLOCK_N / 4);
    unsigned int const a_stride = num_threads / (BLOCK_K / 4);
    unsigned int const b_stride = num_threads / (BLOCK_N / 4);

    // allocate thread-local cache for results in register file
    float Tacc[THREAD_M * THREAD_N] = {0.f,};
    // register cache for As and Bs
    float At[THREAD_M] = {0.f,};
    float Bt[THREAD_N] = {0.f,};

    // outer loop over block tiles
    #pragma unroll
    for (unsigned int bk_idx = 0; bk_idx < k; bk_idx += BLOCK_K) {
        // allocate smem space for the current blocktile
        __shared__ float As[BLOCK_M * BLOCK_K];
        __shared__ float Bs[BLOCK_K * BLOCK_N];

        // load the matrices from global memory to shared memory
        // transpose A at this point
        #pragma unroll
        for (unsigned int offset = 0; offset < BLOCK_M; offset += a_stride) {
            float4 tmp = *reinterpret_cast<float4 const*>(&A[(a_inner_row + offset) * k + a_inner_col * 4]);
            As[(a_inner_col * 4 + 0) * BLOCK_M + a_inner_row + offset] = tmp.x;
            As[(a_inner_col * 4 + 1) * BLOCK_M + a_inner_row + offset] = tmp.y;
            As[(a_inner_col * 4 + 2) * BLOCK_M + a_inner_row + offset] = tmp.z;
            As[(a_inner_col * 4 + 3) * BLOCK_M + a_inner_row + offset] = tmp.w;
        }
        #pragma unroll
        for (unsigned int offset = 0; offset < BLOCK_K; offset += b_stride) {
            *reinterpret_cast<float4*>(&Bs[(b_inner_row + offset) * BLOCK_N + b_inner_col * 4]) =
                *reinterpret_cast<float4 const*>(&B[(b_inner_row + offset) * n + b_inner_col * 4]);
        }
        cta.sync(); // sync to make sure the matrices are loaded

        // advance blocktile
        A += BLOCK_K;       // move blocktile to right
        B += BLOCK_K * n;   // move blocktile to down

        // calculate per-thread results
        #pragma unroll
        for (unsigned int i = 0; i < BLOCK_K; i++) { // dot product loop
            // block into registers
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_M; t++) {
                At[t] = As[i * BLOCK_M + thread_row * THREAD_M + t];
            }
            #pragma unroll
            for (unsigned int t = 0; t < THREAD_N; t++) {
                Bt[t] = Bs[i * BLOCK_N + thread_col * THREAD_N + t];
            }

            #pragma unroll
            for (unsigned int tm = 0; tm < THREAD_M; tm++) {
                #pragma unroll
                for (unsigned int tn = 0; tn < THREAD_N; tn++) {
                    Tacc[THREAD_N * tm + tn] += At[tm] * Bt[tn];
                }
            }
        }
        cta.sync();
    }

    // write out the results
    #pragma unroll
    for (unsigned int tm = 0; tm < THREAD_M; tm++) {
        #pragma unroll
        for (unsigned int tn = 0; tn < THREAD_N; tn += 4) {
            // load C vector into registers
            float4 tmp = *reinterpret_cast<float4*>(&C[(thread_row * THREAD_M + tm) * n + thread_col * THREAD_N + tn]);
            // perform GEMM update in reg
            tmp.x = alpha * Tacc[tm * THREAD_N + tn] + beta * tmp.x;
            tmp.y = alpha * Tacc[tm * THREAD_N + tn + 1] + beta * tmp.y;
            tmp.z = alpha * Tacc[tm * THREAD_N + tn + 2] + beta * tmp.z;
            tmp.w = alpha * Tacc[tm * THREAD_N + tn + 3] + beta * tmp.w;
            // write back
            *reinterpret_cast<float4*>(&C[(thread_row * THREAD_M + tm) * n + thread_col * THREAD_N + tn]) = tmp;
        }
    }
}
```

> vectorize를 적용하면서 bound check 조건이 삭제되었다. 이를 넣어주려면 vectorize할 때 범위를 벗어나는 값은 0으로 처리해주면 될 것 같은데, 위 코드에서는 별도의 처리를 하지 않았다. 따라서 2의 제곱수가 아닌 임의의 수에 대한 동작은 정확한 결과를 내보내지 않는다.

해당 코드에 대한 SASS, PTX 코드는 [link](https://godbolt.org/z/WGYrzdMnf)에서 확인할 수 있다.

이 커널은 약 14 TFLOPs/s의 성능을 가지며, 이전 커널보다 조금 더 향상된 것을 확인할 수 있다. 하지만 프로파일링을 통해 여전히 문제가 있으며 아래와 같은 최적화할 여지가 있다는 것을 보여준다.

- shared-memory bank conflict
- occupancy
- double buffering (CUTLASS [doc](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining) 참조)

각 커널에서 bank conflict 프로파일링 결과는 다음과 같다.
```
void smem_sgemm_kernel<(int)32>(int, int, int, float, const float *, const float *, float, float *), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- -------------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                 0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        34,046,671
    ---------------------------------------------------------------------- -------------------

void smem_1d_blocktiling_sgemm_kernel<(int)64, (int)64, (int)8, (int)8>(int, int, int, float, const float *, const float *, float, float *), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- -------------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                            78,468
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                         7,390,414
    ---------------------------------------------------------------------- -------------------

void smem_2d_blocktiling_sgemm_kernel<(int)64, (int)64, (int)8, (int)8, (int)8>(int, int, int, float, const float *, const float *, float, float *), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- -------------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                       402,653,184
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                       100,663,296
    ---------------------------------------------------------------------- -------------------

void vectorize_sgemm_kernel<(int)64, (int)64, (int)8, (int)8, (int)8>(int, int, int, float, const float *, const float *, float, float *), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- -------------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                       268,435,456
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        33,554,432
    ---------------------------------------------------------------------- -------------------
```

> bank conflict를 해결하기 위해 shared memory에 PADDING을 추가했지만 오히려 bank conflict이 늘어났다. PADDING을 추가하는 것으로는 해결이 안되는 것 같아서 PADDING을 추가한 코드는 따로 첨부하지 않고, 아래의 warptiling을 활용하여 bank conflict가 해결되는지 확인해보자.

# Kernel 6: Warptiling

이전까지 구현한 커널의 loop 구조는 다음과 같다.

<img src="https://siboehm.com/assets/img/CUDA-MMM/Loop_structure.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

이번에 구현할 커널에서는 blocktiling과 threadtiling 루프 사이에 warptiling 계층을 추가한다. 워프를 활용하면 아래의 이유로 인한 성능과 관련이 있다:

- 워프는 SM에서의 스케쥴링 단위이다.
- Shared memory bank conflict는 동일한 워프 내 스레드 사이에서만 발생한다.
- GPU에는 레지스터 캐시가 있어서 더 타이트한 threadtiling으로 레지스터 캐시 locality를 높일 수 있다.

Warptiling을 통해 명시적인 병렬 처리가 가능해진다.

- Blocktiling: 서로 다른 블록이 다른 SM에서 병렬로 실행
- Warptiling: 서로 다른 워프가 다른 워프 스케줄러에 의해 병렬로 실행되며 동일한 워프 스케줄러에서 동시에 실행될 수도 있음
- Threadtiling: (a very limited amount of) 동일한 CUDA 코어에서 병렬로 실행 (= instruction-level parallelism (ILP))

이에 대한 기본 내용은 NVIDIA 에서 발표한 [CUTLASS](https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf) 발표 자료에서 다루고 있으니 이를 참조하면 많은 도움이 된다.

Warptiling이 추가된 계층 구조를 시각화하면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoMKR8%2FbtsyMdLDfEU%2Ff2lqiofavjjck8hxziUiQ0%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

이를 코드로 표현하면 아래와 같이 나타낼 수 있다.
```c++
// fragments used to store data fetched from SMEM
float a_frag[THREAD_M];
float b_grag[THREAD_N];

// accumulator storage
float acc[THREAD_M][THREAD_N];

// GEMM main loop - iterates over the entire K dimension - no unrolling
//                                                       - one iteration of this loop is one "stage"
for (int block_k = 0; block_k < k; block_k += BLOCK_K) {
    // load A and B tiles from global memory and store to SMEM
    ...
    __syncthreads();

    // warp tile structure - iterates over the thread block tile - fully unroll across BLOCK_K
    //                                                           - one iteration of this loop is one "K Group"
    #pragma unroll
    for (int warp_k = 0; warp_k < BLOCK_K; warp_k++) {
        // fetch a_frag and b_frag from SMEM corresponding to k-index
        ...

        // thread tile structure - accumulate an outer product
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_M; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_N; ++thread_x) {
                acc[thread_y][thread_x] += a_frag[y] * b_frag[x];
            }
        }
    }
    __syncthreads();
}
```

앞서 언급했지만 이렇게 tiling을 통한 중첩된 루프를 통해 

- thread blocks
- warps
- CUDA cores

간의 동시성을 향상시키며, shared memory와 register의 memory locality의 이점을 활용한다.

> 생각보다 구현이 까다롭고, [SGEMM](https://github.com/xldrx/maxas.wiki.fixed/raw/master/img/StoreShared128.png)에서 설명하는 것과 같이 여러 최적화 기법들을 적용해야 cuBLAS에 근접하는 성능을 얻을 수 있는 것으로 보인다. 최대한 비슷하게 구현했지만, vectorize를 적용한 것보다 오히려 성능이 떨어지는 경우도 발생하고 있다.
> 
> 이 구현에서 bank conflict를 해결하는 등의 최적화는 아직 적용하지 못했다. 이에 대한 내용은 다시 최적화 방법에 대해 살펴보고, 다른 포스팅으로 다시 보충하도록 할 예정이다.

구현은 다음과 같다. 사용한 템플릿 파라미터는 아래와 같다.

- `<32, 32, 8, 4, 4, 32, 16>`
- `<64, 64, 8, 4, 4, 32, 16>`

```c++
template<int BLOCK_M, int BLOCK_N, int BLOCK_K,
        int THREAD_M, int THREAD_N,
        int WARP_M, int WARP_N
>
__global__
void warptiling_sgemm_kernel(
    int const m, int const n, int const k,
    float alpha,
    float const* A, float const* B,
    float beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    __shared__ float a_smem[BLOCK_K * BLOCK_M];
    __shared__ float b_smem[BLOCK_K * BLOCK_N];
    float a_frag[THREAD_M] = {0.f}, b_frag[THREAD_N] = {0.f}, acc[THREAD_M][THREAD_N] = {0.f};

    // thread, block, warp, and lane identication
    unsigned int const tid = cta.thread_rank();
    unsigned int const bx = cta.group_index().x;
    unsigned int const by = cta.group_index().y;

    // set blocktile to beginning of A's row, B's column, and C
    A += by * BLOCK_M * k;
    B += bx * BLOCK_N;
    C += by * BLOCK_M * n + bx * BLOCK_N;

    // calculate the indices that this thread will load into SMEM
    // - load 32 bytes => 4 elements per thread at each step
    unsigned int const a_inner_row = tid / (BLOCK_K / 4);
    unsigned int const a_inner_col = tid % (BLOCK_K / 4);
    unsigned int const a_row_offset = cta.num_threads() / 2 / (BLOCK_K / 4);
    unsigned int const b_inner_row = (tid - cta.num_threads() / 2) / (BLOCK_N / 4);
    unsigned int const b_inner_col = (tid - cta.num_threads() / 2) % (BLOCK_N / 4);
    unsigned int const b_row_offset = cta.num_threads() / 2 / (BLOCK_N / 4);

    unsigned int a_idx, b_idx;
    if (cta.num_threads() == 64) {
        a_idx = ((tid >> 1) & 7);
        b_idx = (((tid & 0x30) >> 3) | (tid & 1));
    }
    else if (cta.num_threads() == 256) {
        a_idx = (((tid & 128) >> 4) | ((tid >> 1) & 7));
        b_idx = (((tid & 0x70) >> 3) | (tid & 1));
    }

    // GEMM main loop - iterates over the entire K dimensions - no unrolling
    for (int block_k = 0; block_k < k; block_k += BLOCK_K) {
        // load A and B matrics from global memory and store to SMEM
        if (tid < cta.num_threads() / 2) {
            #pragma unroll
            for (unsigned int offset = 0; offset < BLOCK_M; offset += a_row_offset) {
                float4 tmp = *(float4 const*)(&A[(a_inner_row + offset) * k + a_inner_col * 4]);
                a_smem[(a_inner_col * 4 + 0) * BLOCK_M + a_inner_row + offset] = tmp.x;
                a_smem[(a_inner_col * 4 + 1) * BLOCK_M + a_inner_row + offset] = tmp.y;
                a_smem[(a_inner_col * 4 + 2) * BLOCK_M + a_inner_row + offset] = tmp.z;
                a_smem[(a_inner_col * 4 + 3) * BLOCK_M + a_inner_row + offset] = tmp.w;
            }
        }
        else {
            #pragma unroll
            for (unsigned int offset = 0; offset < BLOCK_K; offset += b_row_offset) {
                *(float4*)(&b_smem[(b_inner_row + offset) * BLOCK_N + b_inner_col * 4]) =
                    *(float4 const*)(&B[(b_inner_row + offset) * n + b_inner_col * 4]);
            }
        }
        cta.sync();

        // advance blocktile
        A += BLOCK_K;
        B += BLOCK_K * n;

        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k++) {
            // fetch a_frag and b_frag from SMEM corresponding to k-index
            *(float4*)(a_frag) = *(float4*)(&a_smem[warp_k * BLOCK_M + a_idx * 4]);
            *(float4*)(b_frag) = *(float4*)(&b_smem[warp_k * BLOCK_N + b_idx * 4]);

            #pragma unroll
            for (unsigned int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (unsigned int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        cta.sync();
    }
    
    // write out results
    unsigned int const c_row = a_idx * 4;
    unsigned int const c_col = b_idx * 4;
    #pragma unroll
    for (unsigned int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (unsigned int j = 0; j < THREAD_N; j++) {
            C[(c_row + i) * n + c_col + j] = alpha * acc[i][j] + beta * C[(c_row + i) * n + c_col + j];
        }
    }
}
```

# Results for Each Size

256부터 4096까지 `M=N=K`의 크기를 변경하며 측정한 각 커널의 평균 실행 시간 결과는 다음과 같다.

```
- Results (ms)
                                      Size (M=N=K)     256     384     512     640     768     896    1024    1152    1280    1408    1536    1664    1792    1920    2048    2176    2304    2432    2560    2688    2816    2944    3072    3200    3328    3456    3584    3712    3840    3968    4096
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                            cuBLAS   0.011   0.020   0.029   0.048   0.072   0.101   0.125   0.216   0.248   0.289   0.444   0.521   0.600   0.889   0.842   1.330   1.299   1.563   1.707   2.108   2.383   2.877   3.127   3.620   3.765   4.602   4.806   5.467   5.950   6.617   7.008
                                       naive_sgemm   0.025   0.098   0.171   0.316   0.518   0.823   1.249   1.960   2.732   3.645   4.686   5.971   7.576   9.197  11.144  13.755  16.468  19.392  22.711  26.305  30.320  34.772  40.141  46.061  51.797  57.865  64.117  70.962  78.097  85.717  93.619
                                        smem_sgemm   0.020   0.074   0.128   0.241   0.418   0.616   0.898   1.354   1.813   2.405   3.073   3.931   4.972   6.006   7.344   8.780  10.513  12.275  14.446  16.612  19.199  22.253  26.036  29.795  33.361  37.103  41.384  45.182  49.829  54.719  59.778
           smem_1d_blocktiling_sgemm<64, 64, 8, 8>   0.030   0.043   0.056   0.106   0.190   0.225   0.316   0.456   0.604   0.863   1.088   1.334   1.685   2.102   2.540   2.962   3.609   4.234   4.879   5.603   6.527   7.595   8.578   9.851  10.999  12.226  13.720  15.078  16.446  18.225  20.110
           smem_2d_blocktiling_sgemm<64, 64, 8, 8>   0.038   0.055   0.070   0.091   0.142   0.167   0.215   0.302   0.374   0.532   0.741   0.828   1.035   1.298   1.578   1.889   2.240   2.617   3.033   3.471   4.166   4.665   5.284   6.265   6.764   7.710   8.558   9.336  10.389  11.557  12.811
                     vectorize_sgemm<64, 64, 8, 8>   0.033   0.047   0.061   0.083   0.105   0.134   0.174   0.229   0.282   0.407   0.594   0.671   0.798   0.984   1.205   1.476   1.726   2.005   2.336   2.713   3.197   3.628   4.135   4.750   5.265   5.934   6.596   7.197   7.989   8.861   9.742
                   vectorize_sgemm<128, 128, 8, 8>   0.048   0.068   0.089   0.109   0.136   0.161   0.186   0.303   0.344   0.386   0.637   0.708   0.778   0.997   1.099   1.479   1.592   1.960   2.117   2.608   3.065   3.291   3.978   4.562   4.834   5.481   6.063   6.762   7.429   8.261   9.065
      warptiling_sgemm_kernel<64, 32, 32, 8, 4, 4>   0.028   0.043   0.056   0.087   0.116   0.145   0.194   0.356   0.417   0.512   0.757   0.883   1.096   1.371   1.618   1.978   2.313   2.737   3.159   3.661   4.302   4.858   5.487   6.322   7.136   7.859   8.870   9.805  10.706  12.137  13.511
     warptiling_sgemm_kernel<256, 64, 64, 8, 4, 4>   0.026   0.038   0.049   0.071   0.098   0.131   0.184   0.239   0.334   0.469   0.564   0.707   0.899   1.103   1.342   1.575   1.910   2.236   2.581   2.980   3.454   3.972   4.437   5.038   5.649   6.301   7.065   7.834   8.633   9.553  10.540
```


<br>

# References

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [RTX 3080 Spec](https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621)
- [CUTLASS Doc: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Register Cache: Catching for Warp-Centric CUDA Programs](https://developer.nvidia.com/blog/register-cache-warp-cuda/)
- [SGEMM](https://github.com/NervanaSystems/maxas/wiki/SGEMM)
- [Nvidia Tensor Core-CUDA HGEMM Advanced Optimization](https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85)