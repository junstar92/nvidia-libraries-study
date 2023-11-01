# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Where is the Warp-Level Cache ?](#where-is-the-warp-level-cache-)
- [Caching in Registers Using Shuffle](#caching-in-registers-using-shuffle)
- [Register Cache by Example: 1D Stencil](#register-cache-by-example-1d-stencil)
  - [1D Stencil](#1d-stencil)
  - [Shared Memory Implementation](#shared-memory-implementation)
  - [Register Cache Implementation](#register-cache-implementation)
- [Results](#results)
- [References](#references)

<br>

# Introduction

이번 포스팅에서는 레지스터 캐시(register cache)에 대해서 설명하고 있는 [link](https://developer.nvidia.com/blog/register-cache-warp-cuda/)를 리뷰한다. 레지스터 캐시는 한 워프(warp) 내에서 스레드(threads)에 대한 virtual caching layer를 구현하는 최적화 기법이며, NVIDIA GPU의 shuffle primitive 위에 구현된 소프트웨어 추상화이다. 이는 shared memory를 사용하여 스레드 입력을 캐싱하는 커널을 최적화하는데 도움이 된다. 이 기법을 사용하면 데이터는 각 워프의 스레드에 있는 레지스터에 분산되고, shared memory에 대한 액세스는 shuffle을 사용하여 다른 스레드의 레지스터에 대한 액세스로 바뀌므로 성능상 상당한 이점을 얻을 수 있다.

> 레지스터 캐시는 shared memory를 register로 대체하는 최적화라고 말할 수 있다.

포스팅에서는 1D-Stencil을 계산하는 간단한 커널을 통해 레지스터 케시 추상화를 구현하고 사용하는 방법에 대해서 살펴본다.

# Where is the Warp-Level Cache ?

커널은 NVIDIA의 세 가지 메모리 계층인 global memory, shared memory, registers에 데이터를 저장할 수 있다. 이러한 계층은 크기, 성능, 공유 범위 관점에서 효과적으로 계층 구조를 형성한다. 크기가 가장 큰 global memory는 커널 내 모든 스레드에서 공유된다. Global memory보다 크기는 작지만 훨씬 더 빠른 shared memory는 단일 스레드 블록 내의 모든 스레드에서 공유된다. 그리고, 가장 크기가 작으며 가장 빠른 레지스터는 각 스레드 내에서만 사용되는 리소스이며 서로 다른 스레드에 non-visible이다.

이러한 메모리 계층을 실행 계층에서의 캐시로 볼 수 있다. Shared memory의 경우, 스레드 블록 내 스레드에 대한 캐시 역할을 담당라며, 레지스터는 단일 스레드의 데이터 캐시 역할을 담당하는 것과 같다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/paper4-500x279.jpg" height=250px style="display: block; margin: 0 auto; background-color:white"/>

이를 그림으로 표현하면 위와 같다. 위 그림에서 볼 수 있듯이 커널, 스레드 블록, 각 스레드에 대한 캐시 레이어는 명시적으로 존재하지만, 단일 워프에 대한 캐시 레이어는 없다. 따라서, 워프 단위로 설계되는 커널에서 워프 입력을 캐시하는 하드웨어 메모리 계층이 존재하지 않는다.

# Caching in Registers Using Shuffle

CUDA 케플러 아키텍처에서는 워프 내 커뮤니케이션이 가능하도록 SHFL (shuffle) instruction을 도입했다. 프리미티브 함수인 `shfl_sync(m, r, t)`는 동일 워프에 존재하는 다른 스레드 `t`가 공유하는 값을 읽는 동안 호출한 스레드의 레지스터 `r`에 저장된 값을 공유할 수 있다. 여기서 `m`은 워프 내에서 커뮤니케이션에 참여하는 스레드에 대한 32-bit mask이다. 아래 그림은 `shfl_sync()`의 동작을 보여준다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/paper5-625x157.jpg" height=150px style="display: block; margin: 0 auto; background-color:white"/>

워프 내 스레드 간 데이터 공유에서 레지스터를 사용하는 이유는 다음과 같다.

1. 레지스터의 bandwidth는 shared memory보다 크고, latency는 더 작다.
2. `shfl_sync()`를 사용하면 스레드의 writing과 reading 간의 `__syncthreads`와 같은 memory fence를 통한 스레드 블록 동기화가 필요없게 된다. Shared memory를 사용하면 스레드 블록 내 동기화가 필요하다.
3. 레지스터 파일의 크기는 256KB이다 (일반적으로 shared memory를 64KB~100KB).

다만, 셔플을 사용하는 것은 꽤 까다롭고 복잡하다. 특히 shared memory를 사용하도록 이미 커널이 구현된 경우에 셔플을 사용하려면 상당한 수정이 필요할 수 있다.

> 여기서 설명하는 레지스터 캐시 기법은 shraed memory access를 shuffle로 대체하여 커널을 최적화하는 방향을 제시한다. 일반적으로 shared memory를 사용하여 커널의 입력을 캐시하는 케이스를 대상으로 한다.

# Register Cache by Example: 1D Stencil

## 1D Stencil

먼저 1D k-stencil의 정의는 다음과 같다.

> **Definition. 1D k-stencil**: 크기가 `n`인 배열 `A`가 주어졌을 때, A의 k-stencil은 크기가 `n - 2k`인 배열 `B`이다.

$$ B[i] = (A[i] + ... + A[i + 2k]) / (2k + 1) $$

예를 들어, 아래와 같이 배열 `A`가 주어졌을 때,

- `A = [0, 1, 2, 3, 4, 5, 6, 7]`

배열 `A`의 1-stencil인 행렬 `B`는 다음과 같다.

- `B = [1, 2, 3, 4, 5, 6]`

여기서 행렬 `B`의 각 요소는 아래와 같이 계산된다.

- `B[0] = (A[0] + A[1] + A[2]) / 3 = (0 + 1 + 2) / 3 = 1`
- `B[1] = (A[1] + A[2] + A[3]) / 3 = (1 + 2 + 3) / 3 = 2`
- ...
- `B[5] = (A[5] + A[6] + A[7]) / 3 = (5 + 6 + 7) / 3 = 6`

1D k-stencil를 계산하려면, 각 입력 요소를 `2k + 1`번 read해야 한다 (margin 제외). 즉, 입력이 여러 번 사용되므로 데이터 재사용을 활용하기 위해서 캐시를 반드시 사용해야 한다.

## Shared Memory Implementation

먼저 shared memory를 사용한 커널 구현부터 살펴보자. 구현은 다음 단계를 따른다.

1. Global memory로부터 입력을 shared memory로 복사
2. `__syncthreads()`를 호출하여 모든 입력이 shared memory로 복사될 때까지 대기
3. 하나의 출력 요소 계산
4. 결과값을 다시 global memory에 저장

위 단계를 그림으로 표현하면 다음과 같다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/paper6.jpg" height=250px style="display: block; margin: 0 auto; background-color:white"/>

커널 구현은 아래와 같다.

```c++
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
        s[blockDim.x + local_id] = A[blockDim.x + global_id];
    }

    __syncthreads(); // sync before reading from shared memory

    // each thread computes a single output
    if (global_id < size - 2) {
        B[global_id] = (smem[local_id] + smem[local_id + 1] + smem[local_id + 2]) / 3;
    }
}
```

## Register Cache Implementation

이제 레지스터 캐시를 사용한 구현을 살펴보자.

구현하기에 앞서 먼저 각 워프에 필요한 입력이 무엇인지 생각해보자. 워프 내에서 therad 0이 계산하는 출력 요소의 (global)인덱스를 `i`라고 한다면, 이 워프는 `i`, `i+1`, ..., `i+31`의 출력 요소를 계산하게 된다. 따라서, 이 워프의 입력 요소는 입력 배열에서 `i-1`, `i`, `i+1`, ..., `i+32` 인덱스의 요소라는 것을 알 수 있다.

우리는 단일 워프 내에서 각 레지스터 간 입력을 분배해야 한다. 이 예제에서는 round-robin distribution을 사용하여 스레드 간 입력 배열을 분배한다. 이 방법에서 `input[i]`는 thread `j`에 할당되며, 이때, `j = i % 32`이다. 따라서, thread 0과 thread 1은 각각 두 개의 요소를 저장하고, 나머지 스레드들은 오직 하나의 요소만 저장하게 된다. 각 스레드에서 캐시되는 첫 번째 요소는 `rc[0]`이라고 표기하고, 두 번째 요소는 `rc[1]`이라고 표기한다.

워프 내 스레드들에 분배되는 입력 값을 표로 나타내면 아래와 같다.
|Thread| T0 | T1 | T2 | T3 | ... | T31 |
|-|-|-|-|-|-|-|
|**rc**|`input[0]`, `input[32]`|`input[1]`, `input[33]`|`input[2]`|`input[3]`|...|`input[31]`|

그럼 이제 communication과 compute 단계를 살펴보자.

커뮤니케이션 단계에서는 스레드들이 레지스터 캐시에 효과적으로 액세스하게 된다. 이 단계에서 각 스레드는 캐시에서 읽은 값을 사용하여 지역적으로 일부 산술 연산과 논리 연산을 수행한다.

레지스터 캐시에는 두 가지 기술적 요소가 있는데 이는 커뮤니케이션 단계를 쉽게 설계할 수 있도록 워프의 스레드에서 사용된다.

1. `Read(src_tid, remote_reg)`: `src_tid` 스레드에서 remote variable인 `remote_reg`에 저장된 데이터를 read 한다.
2. `Publish(local_reg)`: `local_reg` 변수에 저장된 local 데이터를 publish 한다.

각 커뮤니케이션 단계는 하나 이상의 위 프리미티브들로 구성되며, 한 스레드가 `Read`를 수행하려면 다른 스레드가 local 레지스터에 저장된 requested data를 `Publish`해야 한다.

우리가 현재 구현하는 1-stencil에서는 3단계의 커뮤니케이션이 필요하다. 이는 하나의 출력 요소를 계산할 때 3개의 요소가 필요하기 때문이다. [link](https://developer.nvidia.com/blog/register-cache-warp-cuda/)에서는 각 단계에서 필요한 `Read`, `Publish` 연산을 잘 보여주고 있다. 만약 워프에 4개의 스레드만 존재한다면, 각 커뮤니케이션 단계는 아래와 같이 표현된다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/comm_paper_1-362x664.jpg" height=500px style="display: block; margin: 0 auto; background-color:white"/>

각 단계에서 스레드에 필요한 데이터를 셔플 연산을 통해서 얻는다. 예를 들어, 첫 번째 단계에서 각 스레드에 필요한 요소는

- `T0` - `A[0]`
- `T1` - `A[1]`
- `T2` - `A[2]`
- `T3` - `A[3]`

이 되고, 두 번째 커뮤니케이션 단계에서는

- `T0` - `A[1]`
- `T1` - `A[2]`
- `T2` - `A[3]`
- `T3` - `A[4]`(from `T0`)

마지막 세 번째 단계는

- `T0` - `A[2]`
- `T1` - `A[3]`
- `T2` - `A[4]`(from `T0`)
- `T3` - `A[5]`(from `T1`)

이 된다.

각 커뮤니케이션 단계가 한 번 완료될 때마다 각 스레드는 global memory에 출력값을 저장하기 위해 `ac`라는 레지스터에 누적한다.

<br>

CUDA에서는 `Read`, `Publish` 프리미티브는 제공하지 않는다. 대신 두 연산을 합친 셔플 프리미티브를 제공한다. 이 프리미티브가 바로 `__shfl_sync()`이다.

`__shfl_sync()`를 사용하여 구현한 커널 함수는 아래와 같다.

```c++
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
```

# Results

아래 결과는 NVIDIA 포스트에서 실험한 결과이다. k 값을 증가함에 따라 shared memory 구현보다 레지스터 캐시 구현가 더 빠르다는 것을 보여준다. k 값이 작으면 데이터 재사용이 작기 때문에 속도 향상이 적고, k가 크면 레지스터 캐시에 의한 데이터 재사용이 증가하여 속도가 빠르다고 설명하고 있다 (GTX 1080 and CUDA 9).

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/10/chart-1-e1507780043676.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

하지만 실제로 구현했을 때, 위와 같은 성능 비교 결과가 측정되지는 않았다 (RTX 3080 and CUDA 11.8).

```
$ k_stensil 2
> 2-Stencil - size: 134217728
k_stencil<<<131072, 1024, 4104>>>: 2.507 ms
2 3 4 5 6 7 8 9 10 11 12 13 14 11 9 6 4 2 3 4 5 6 7 8 9 10 11 12 13 14 11 9 
k_stencil_with_rc<<<131072, 1024>>>: 2.249 ms
2 3 4 5 6 7 8 9 10 11 12 13 14 11 9 6 4 2 3 4 5 6 7 8 9 10 11 12 13 14 11 9

$ k_stensil 5
> 5-Stencil - size: 134217728
k_stencil<<<131072, 1024, 4116>>>: 2.634 ms
5 6 7 8 9 10 11 10 9 9 8 8 7 7 6 6 5 5 6 7 8 9 10 11 10 9 9 8 8 7 7 6 
k_stencil_with_rc<<<131072, 1024>>>: 2.808 ms
5 6 7 8 9 10 11 10 9 9 8 8 7 7 6 6 5 5 6 7 8 9 10 11 10 9 9 8 8 7 7 6
```

생각보다 레지스터 캐시에 의한 효과 상승을 누리지 못했고, 1-stencil에서도 배열의 크기가 매우 커야 레지스터 캐시로 약간의 성능 이득을 볼 수 있다는 결과를 얻었다.

> 전체 구현 코드는 [k_stencil.cu](/cuda/code/k_stencil/k_stencil.cu)에서 확인할 수 있다.

결론적으로 생각보다 성능 향상이 크지 않고, 이러한 기법을 적용할 수 있는 케이스도 그리 많지 않다고 생각된다.

<br>

# References

- [Register Cache: Catching for Warp-Centric CUDA Programs](https://developer.nvidia.com/blog/register-cache-warp-cuda/)
- [Fast Multiplication in Binary Fields on GPUs via Register Cache](https://pdfs.semanticscholar.org/67f8/e4f254e0ed1f68fac04b588955b87c438aa2.pdf)