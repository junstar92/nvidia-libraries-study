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
|6|Warptiling SGEMM| 8.337 | 15.358 | 89.19% |

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

> bank conflict를 해결하기 위해 shared memory에 PADDING을 추가했지만 오히려 bank conflict이 늘어났다. 이는 코드 상에서 shared memory에 액세스하는 패턴이 원인으로 보이며, 아래의 warp tiling 패턴으로 구현한 버전을 통해 bank conflict를 해결할 수 있는지 확인해보자.

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

각 계층 구조를 천천히 다시 한 번 살펴보자.

먼저 결과 행렬 기준으로 행렬을 스레드 블록 단위로 구분한다. 그림으로 표현하면 아래와 같은데, 초록색으로 표시된 하나의 블록을 하나의 스레드 블록이 처리하게 된다. 이 구조에서 행렬 데이터는 global memory에 상주한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNftCQ%2FbtszekXex1B%2Fo2OGzvJgsFc4YzgE6qQ2Fk%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

> 일반적으로 스레드 블록의 크기는 `64 x 64` 또는 `128 x 128`로 설정되며, 스레드 블록의 크기에 따라서 하나의 스레드 블록에 할당되는 스레드 갯수는 다르다.

하나의 스레드 블록의 구조는 다음과 같다. 그리고, 하나의 스레드 블록이 처리하는 데이터는 global memory에서 shared memory로 복사한 다음, 이를 사용하여 계산하게 되는 것까지는 이전과 동일하다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FczO2Un%2Fbtsy9OS7Zq3%2FP0SskkcQ6rowVDoumbk2d0%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

하지만 내부에서 연산을 분할하는 방법이 기존의 커널 구현과 다르다. 위 그림을 보면 하나의 스레드 블록이 `2 x 4`개의 블록으로 나누어진 것을 볼 수 있다. 이 세부 블록 하나가 바로 워프 하나가 담당하는 블록이 된다. 즉, 위 그림에서는 하나의 스레드 블록이 8개의 워프가 각각의 서브 블록 계산을 담당하게 된다. 하나의 워프에는 32개의 스레드가 있으므로, 위 그림의 경우에는 `32 x 8 = 256`개의 스레드가 하나의 스레드 블록에 속하게 된다.

> Warp-level GEMM의 목적은 CUDA execution model 내에서 warp-level 병렬화에 매핑하기 위함이다. GPU 내 SM의 warp-scheduler에 의해서 워프 단위로 실행한다. 아래 구현에서는 CUDA core에 이슈(issue)된 thread-level의 행렬 연산을 구현했지만, 이러한 구조를 통해 Tensor core를 사용하도록 `mmm.sync` 또는 `wmma` 명령어를 사용할 수 있다.

다음으로 하나의 워프 블록의 구조를 세부적으로 살펴보자.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb2fUdt%2FbtszcFnhI6D%2FxojK2lDunggrBAAR6KscLK%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

하나의 워프 블록 내부 구조는 위와 같다. 여기서 워프 블록은 `16 x 8`개의 서브 블록으로 나뉘며, 여기서 하나의 블록을 일반적으로 타일(tile)이라고 지칭한다. 그리고 하나의 타일은 `4 x 4` 행렬로 구성된다. 그림을 자세히 살펴보면, 하나의 워프 타일을 크게 4개로 분할한 것을 볼 수 있으며 분할된 블록들이 각각 `8 x 4`개의 타일로 이루어져 있는 것을 볼 수 있다. 그 이유는 워프 내 스레드의 갯수는 32개이고, 각 스레드가 `8 x 4`개의 타일 중 하나를 담당하여 계산하기 위함이다. 여기서 `8 x 4` 크기의 블록이 4개가 있는데, 32개의 스레드가 4개의 `8 x 4` 크기의 블록 계산을 담당하게 되며, 따라서, 하나의 스레드는 4개의 타일을 계산하게 된다.

각 스레드가 처리하는 타일을 확대하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFhWqx%2FbtszcIqJBxA%2FW8vApJmKoa5j5p7U5Kw4ok%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

가장 낮은 수준의 블록이며 각 스레드가 특정 갯수의 요소를 처리하게 되는데, 위 경우에는 하나의 스레드가 `8 x 8 = 64`개의 요소를 계산하게 된다. 각 스레드가 계산에 사용하는 행렬 A, B의 요소는 shared memory로부터 레지스터로 이동된다. 각 스레드들은 서로 다른 스레드의 레지스터에 액세스할 수 없기 때문에 하나의 스레드에서 행렬 요소를 계산할 때 레지스터를 최대한 재사용할 수 있도록 2차원 구조의 타일을 사용하게 된다. 여기서는 shared memory에 상주하는 A 행렬의 8개의 요소와 B 행렬의 8개의 요소를 레지스터로 가져오고, 레지스터로 가져온 요소 값을 이용하여 16개의 외적(outer product)값을 누적하게 된다.


위 과정을 코드로 표현하면 아래와 같이 나타낼 수 있다.
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

Warptiling을 활용한 SGEMM 커널의 구현은 다음과 같다.

```c++
#define WARP_SIZE       32
#define WARP_TILE_M     16
#define WARP_TILE_N     8
#define WARP_THREAD_M   8
#define WARP_THREAD_N   4
#define THREAD_TILE_M   8
#define THREAD_TILE_N   8
#define LANE_LAYOUT     2
#define LANE_M          4
#define LANE_N          4
#define PADDING         4

template<int NUM_THREADS, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int WARP_M, int WARP_N, int WARP_K>
__global__
void warptiling_sgemm_kernel(
    int const M, int const N, int const K,
    float alpha,
    float const* A, float const* B,
    float beta,
    float* C
)
{
    cg::thread_block cta = cg::this_thread_block();

    int constexpr WARP_NUM_ROW = (BLOCK_M / WARP_M);
    int constexpr WARP_NUM_COL = (BLOCK_N / WARP_N);
    int const thread_idx = cta.thread_rank();
    // threadblock-level indices
    int const block_idx_m = cta.group_index().y;
    int const block_idx_n = cta.group_index().x;
    // warp-level indices of each thread
    int const lane_idx = thread_idx % WARP_SIZE;
    int const warp_idx = thread_idx / WARP_SIZE;
    int const tb_warp_idx_m = warp_idx % (WARP_NUM_ROW);
    int const tb_warp_idx_n = warp_idx / (WARP_NUM_ROW);
    int const warp_tile_idx_m = ((lane_idx >> 3) << 1) + (lane_idx & 1);
    int const warp_tile_idx_n = ((lane_idx & 7) >> 1);
    int const tb_tile_idx_m = warp_tile_idx_m + tb_warp_idx_m * WARP_TILE_M;
    int const tb_tile_idx_n = warp_tile_idx_n + tb_warp_idx_n * WARP_TILE_N;

    // set blocktile to beginning of A's row and B's column
    A += block_idx_m * BLOCK_M * K;
    B += block_idx_n * BLOCK_N;

    // allocate smem space for the threadblock
    __shared__ float a_smem[BLOCK_K][BLOCK_M + PADDING];
    __shared__ float b_smem[BLOCK_K][BLOCK_N];

    // allocate thread-local cache for results in register file
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.f,};
    // register cache for As and Bs
    float a_frag[THREAD_TILE_M] = {0.f,};
    float b_frag[THREAD_TILE_N] = {0.f,};

    // element indices for writing smem from global memory
    int const a_tb_idx_m = thread_idx / BLOCK_K;
    int const a_tb_idx_k = thread_idx % BLOCK_K;
    int const b_tb_idx_k = thread_idx / BLOCK_N;
    int const b_tb_idx_n = thread_idx % BLOCK_N;

    // GEMM main loop - iterates over the entire K dimension - no unrolling
    for (unsigned int block_k = 0; block_k < K; block_k += BLOCK_K) {
        // load A and B tiles from global memory and store to SMEM
        #pragma unroll
        for (int k = 0; k < (BLOCK_K * BLOCK_M / NUM_THREADS); k++) {
            a_smem[a_tb_idx_k][k * (NUM_THREADS / BLOCK_K) + a_tb_idx_m] = A[(k * (NUM_THREADS / BLOCK_K) + a_tb_idx_m) * K + a_tb_idx_k];
        }
        #pragma unroll
        for (int k = 0; k < (BLOCK_K * BLOCK_N / NUM_THREADS); k++) {
            b_smem[k * (NUM_THREADS / BLOCK_N) + b_tb_idx_k][b_tb_idx_n] = B[(k * (NUM_THREADS / BLOCK_N) + b_tb_idx_k) * N + b_tb_idx_n];
        }
        cta.sync();

        // advance blocktile
        A += BLOCK_K;
        B += BLOCK_K * N;

        // warp tile structure - iterates over the thread block tile - fully unroll across BLOCK_K
        #pragma unroll
        for (int warp_k = 0; warp_k < BLOCK_K; warp_k += (BLOCK_K / WARP_K)) {
            // fetch a_frag and b_frag from SMEM corresponding to k-index
            #pragma unroll
            for (int m = 0; m < LANE_LAYOUT; m++) {
                *reinterpret_cast<float4*>(&a_frag[m * LANE_M]) = 
                    *reinterpret_cast<float4*>(&a_smem[warp_k][(tb_tile_idx_m + m * WARP_THREAD_M) * LANE_M]);
            }
            #pragma unroll
            for (int n = 0; n < LANE_LAYOUT; n++) {
                *reinterpret_cast<float4*>(&b_frag[n * LANE_N]) =
                    *reinterpret_cast<float4*>(&b_smem[warp_k][(tb_tile_idx_n + n * WARP_THREAD_N) * LANE_N]);
            }

            // mma in thread tile structure
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        cta.sync();
    }

    // set warptile to beginning of C's row and column
    C += (block_idx_m * BLOCK_M + tb_warp_idx_m * WARP_M) * N + block_idx_n * BLOCK_N + tb_warp_idx_n * WARP_N;
    // write out the results
    #pragma unroll
    for (int m = 0; m < LANE_LAYOUT; m++) {
        #pragma unroll
        for (int n = 0; n < LANE_LAYOUT; n++) {
            #pragma unroll
            for (int k = 0; k < LANE_M; k++) {
                *reinterpret_cast<float4*>(&C[((warp_tile_idx_m + m * WARP_THREAD_M) * LANE_M + k) * N + (warp_tile_idx_n + n * WARP_THREAD_N) * LANE_N]) = 
                    *reinterpret_cast<float4*>(&accum[m * LANE_M + k][n * LANE_N]);
            }
        }
    }
}
```

이전 커널에 비해서 약간의 성능 향상을 달성한 것을 확인할 수 있다.


# Results for Each Size

256부터 4096까지 `M=N=K`의 크기를 변경하며 측정한 각 커널의 평균 실행 시간 결과는 다음과 같다.

```
- Results (ms)
                                      Size (M=N=K)     256     384     512     640     768     896    1024    1152    1280    1408    1536    1664    1792    1920    2048    2176    2304    2432    2560    2688    2816    2944    3072    3200    3328    3456    3584    3712    3840    3968    4096
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                            cuBLAS   0.011   0.020   0.026   0.044   0.073   0.099   0.121   0.207   0.234   0.279   0.435   0.506   0.596   0.873   0.830   1.320   1.258   1.554   1.668   2.088   2.340   2.844   3.099   3.521   3.729   4.580   4.802   5.515   6.047   6.654   6.960
                                       naive_sgemm   0.025   0.098   0.152   0.284   0.519   0.814   1.240   1.921   2.707   3.613   4.656   5.905   7.488   9.108  11.024  13.583  16.320  19.224  22.526  26.051  30.021  34.511  39.286  44.906  51.475  57.795  64.310  71.266  78.464  86.010  93.919
                                        smem_sgemm   0.020   0.074   0.114   0.226   0.414   0.606   0.892   1.336   1.801   2.396   3.071   3.911   4.945   6.002   7.349   8.762  10.502  12.250  14.442  16.592  19.129  21.889  24.956  28.981  33.319  36.937  41.469  45.712  50.173  55.263  60.330
           smem_1d_blocktiling_sgemm<64, 64, 8, 8>   0.030   0.043   0.050   0.106   0.190   0.224   0.310   0.451   0.597   0.856   1.076   1.315   1.671   2.079   2.516   2.906   3.581   4.166   4.848   5.538   6.481   7.484   8.357   9.575  10.954  12.273  13.731  15.173  16.531  18.268  20.159
           smem_2d_blocktiling_sgemm<64, 64, 8, 8>   0.038   0.055   0.063   0.092   0.141   0.165   0.212   0.302   0.370   0.521   0.730   0.809   1.016   1.264   1.553   1.859   2.200   2.563   2.963   3.393   4.091   4.641   5.196   6.047   6.743   7.647   8.597   9.319  10.376  11.555  12.778
                  vectorize_sgemm<64, 64, 8, 8, 8>   0.033   0.047   0.054   0.083   0.104   0.132   0.171   0.225   0.281   0.402   0.587   0.665   0.795   0.965   1.201   1.456   1.705   1.982   2.314   2.666   3.163   3.589   4.002   4.653   5.191   5.888   6.581   7.206   8.007   8.884   9.749
                vectorize_sgemm<128, 128, 8, 8, 8>   0.048   0.068   0.079   0.108   0.136   0.160   0.184   0.299   0.339   0.381   0.631   0.697   0.774   0.979   1.082   1.443   1.588   1.922   2.095   2.555   3.013   3.264   3.825   4.442   4.789   5.458   6.048   6.809   7.498   8.282   9.081
     warptiling_sgemm_kernel<64, 64, 8, 64, 32, 8>   0.030   0.043   0.050   0.074   0.109   0.130   0.154   0.214   0.250   0.365   0.540   0.596   0.725   0.906   1.091   1.350   1.569   1.839   2.122   2.461   2.947   3.320   3.743   4.338   4.835   5.536   6.157   6.755   7.549   8.354   9.232
   warptiling_sgemm_kernel<128, 128, 8, 64, 32, 8>   0.043   0.059   0.072   0.099   0.124   0.145   0.171   0.284   0.334   0.368   0.598   0.662   0.728   0.967   1.041   1.398   1.524   1.850   2.017   2.474   2.934   3.145   3.686   4.325   4.596   5.270   5.887   6.544   7.272   7.998   8.762
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