# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [The Parallel Reduction Problem](#the-parallel-reduction-problem)
- [Divergence in Parallel Reduction](#divergence-in-parallel-reduction)
- [Improving Divergence in Parallel Reduction](#improving-divergence-in-parallel-reduction)
- [Reducing with Interleaved Pairs](#reducing-with-interleaved-pairs)
- [References](#references)

<br>

# Intro

일반적으로 스레드 인덱싱은 control flow에 영향을 준다. [Understanding Warp Execution](/cuda-study/06_understanding_warp_execution.md)에서 살펴봤듯이 warp 내에서의 conditional execution은 warp divergence를 발생시켜 커널의 성능 저하를 일으키게 된다. 이런 경우, data access pattern을 재정렬하여 warp divergence를 줄이거나 아예 없앨 수 있다. 이번 포스팅에서는 parallel reduction 문제를 통해 branch divergence를 어떻게 해결할 수 있는지 알아본다.

# The Parallel Reduction Problem

N개의 정수의 합을 계산한다고 하면, 순차 코드로 다음와 같이 쉽게 구현할 수 있다.
```c++
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += arr[i];
}
```

N이 너무 크다면, 병렬로 처리하는 것이 빠를 수 있다. 특히, 덧셈은 교환 법칙과 결합 법칙이 성립하기 때문에 어떠한 순서로 더해도 그 결과는 같다. 따라서, parallel addition은 다음과 같은 방식으로 가능하다.

1. input vector를 더 작은 chunks로 분할
2. 스레드들이 각 chunk의 부분합을 계산
3. 각 chunk의 부분합을 더해 최종합 계산

Parallel addition의 일반적인 방법은 iterative pairwise로 구현하는 것인데, 이 방법에서 스레드는 한 쌍의 요소를 더해 하나의 부분 결과를 구한다. 계산된 부분 결과는 original input vector에 in-place로 저장된다. input에 새롭게 저장된 값들은 다음 iteration에서 새로운 input이 된다. 매 반복마다 input의 수는 절반으로 줄어들기 때문에, output vector의 길이가 1이될 때 마지막 합이 계산된다.

매 반복마다 in-place로 저장되는 output 요소의 위치에 따라서 pairwise parallel sum 구현은 다음의 두 가지 타입으로 분류될 수 있다.

- **Neighbored pair** : 바로 옆에 이웃하는 요소끼리 한 쌍을 이룸
- **Interleaved pair** : 주어진 stride만큼 떨어진 요소끼리 한 쌍을 이룸

<div>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbfxiht%2FbtrYY765slz%2FkFnPobePhRPzkS3A45XFXk%2Fimg.png" height=200px style="display: inline;"/>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCFzOo%2FbtrY2naKVCw%2FhjmgrbXJoTmfpekorlkHk1%2Fimg.png" height=200px style="display: inline;"/>
</div>

위의 그림에서 왼쪽이 neighbored pair 구현이고, 오른쪽이 interleaved pair 구현이다. Neighbored pair 구현에서 각 스레드는 매 반복마다 인접한 두 개의 요소를 취해 하나의 부분합을 계산한다. 반면, interleaved pair 구현에서 각 스레드는 매 반복마다 입력 길이의 절반 크기만큼 떨어진 위치의 두 요소를 취해 부분합을 계산하게 된다.

예를 들어, interleaved pair 방법을 재귀를 사용하여 구현하면 아래와 같이 구현할 수 있다.
```c++
int recursiveReduce(int* data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}
```

위 코드는 덧셈에 대해 구현되었지만, 교환/결합 법칙이 성립되는 모든 연산에 대해서 동일하게 구현할 수 있다. 예를 들어, 최댓값을 구하는 경우, 위 코드에서 '+' 연산을 `max` 연산으로만 바꿔주면 된다. 최솟값, 평균, 곱셈 또한 마찬가지다.

> Vector에 대해 교환 법칙과 결합 법칙이 성립하는 연산을 수행하는 문제를 일반화한 것을 **reduction** problem이라고 부른다. 그리고 **parallel reduction** 은 이 연산을 병렬로 실행하는 것이다.

<br>

# Divergence in Parallel Reduction

> 전체 코드는 [reduce_integer.cu](/code/cuda/reduce_integer/reduce_integer.cu)를 참조

다양한 방법의 parallel reduction 구현을 살펴볼텐데, 먼저 아래 그림과 같은 neighbored pair 방법을 살펴보자. 여기서 각 스레드들은 바로 인접한 두 요소를 더해 하나의 부분합을 계산한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbTfo9e%2FbtrY1nhA9Ym%2F4rNCTSxplMnf5kOZXxo1PK%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

위 그림과 같이 동작하는 커널을 구현해보자. 먼저 각 스레드는 두 개의 인접한 요소를 더해 부분합을 생성한다. 커널 내에서는 두 개의 global memory를 사용하게 되는데, 하나는 reduce 연산을 수행할 input array이고 다른 하나는 각 스레드 블록에서 계산한 부분합을 저장할 input보다는 크기가 작은 output array이다. 각 스레드 블록은 독립적으로 input array의 일부분을 연산하게 된다.

구현은 다음과 같다.
```c++
__global__
void reduceNeighbored(int* g_in, int* g_out, unsigned int const n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            in[tid] += in[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

위 커널 구현을 살펴보면, 한 번의 루프가 한 번의 reduction이라는 것을 알 수 있다. Reduction 연산은 in-place로 수행되며, 각 단계에서 부분합은 input array에 저장된다. 눈여겨 볼 부분 중 하나는 `__syncthreads()`를 사용했다는 것이다. 이 함수는 CUDA 내장 함수이며, 같은 스레드 블록 내의 모든 스레드들이 다음 루프를 진행하기 전에 현재 단계에서의 계산 결과를 저장하도록 해준다 (스레드 블록 내의 모든 스레드들이 `__syncthreads()`에 도달할 때까지 각 스레드들이 대기한다). 마지막 루프를 수행하고나면, 전체 스레드 블록에 대한 부분합이 계산되며, 커널 마지막 부분에 이 값을 저장한다. 이 과정을 시각적으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FehlJcp%2FbtrYXkr43A3%2FmuRzLRaRKPe5gnnBUekhUk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

이렇게 작성한 코드([reduce_integer.cu](/code/cuda/reduce_integer/reduce_integer.cu))를 가지고 컴파일 후, 실행해보면 아래와 같은 출력을 얻을 수 있다. 입력의 크기는 16M(16,777,216 elements)이고, 사용된 스레드 블록(1D)의 크기는 512로 설정했다.

```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce          elapsed 21.3719 ms    cpu sum: 2139353471
gpu Neighbored      elapsed 0.5575 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
```

이 결과를 베이스로 사용하여, 아래 내용을 통해 어떻게 parallel reduction을 더 개선할 수 있는지 살펴보자.

<br>

# Improving Divergence in Parallel Reduction

방금 구현한 `reduceNeighbored` 커널 구현에서 사용되는 아래 조건문을 살펴보면,

```c++
if ((tid % (2 * stride)) == 0)
```

짝수 번째 ID를 갖는 스레드만 `true`가 되기 때문에 **warp divergence** 를 발생시킨다는 것을 알 수 있다 ([Understanding warp execution](/cuda-study/06_understanding_warp_execution.md) 참조).

이 커널의 첫 번째 반복에서 모든 스레드들이 스케쥴링되지만, 오직 짝수 번째의 스레드만(전체의 1/2) 조건문의 body를 수행하게 된다. 두 번째 반복에서도 여전히 모든 스레드들이 스케쥴링되지만, 전체 스레드들 중 1/4만 활성화된다. 결과적으로 warp divergence를 발생시키게 된다.

Warp divergence는 간단히 각 스레드들이 데이터를 인덱싱하는 방법을 바꿔서 개선할 수 있다. 이는 짝수 번째 스레드들이 부분합을 구하는 것이 아닌 인접한 스레드들이 부분합을 구하도록 강제하면 되는데, 그림으로 나타내면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FZp98O%2FbtrYVhh7fiN%2FDxUf6VJID3chLMOY6T4Fpk%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

기존에는 thread 0, thread 2, thread 4, ...의 스레드들이 부분합을 계산했지만, 위와 같이 변경하면 thread 0, thread 1, thread 2, ...의 스레드들이 부분합을 계산하게 된다.

이 방식의 커널 구현은 다음과 같다.
```c++
__global__
void reduceNeighboredLess(int* g_in, int* g_out, unsigned int const n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    int index;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        index = 2 * stride * tid;

        if (index < blockDim.x)
            in[index] += in[index + stride];

        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

이 커널에서 아래 구문을 통해 각 스레드가 액세스하는 데이터 인덱스를 설정해준다.
```c++
index = 2 * stride * tid;
```
`stride`는 2의 배수이므로, `if (index < blockDim.x)` 조건을 통해 스레드 블록의 처음 절반만 사용하도록 해준다.

스레드 블록의 크기가 512 threads이므로, 첫 번째 반복에서는 처음 8개의 warp가 수행되고 나머지 8개의 warp는 아무것도 수행하지 않는다. 두 번째 반복에서는 처음 4개의 warp만 reduction을 수행하고 나머지 12개의 warp는 아무것도 수행하지 않는다. 결과적으로 모든 divergence가 사라진 것은 아니다. 하지만, 반복 단계에서 입력 데이터의 수가 warp size(32)보다 작아지는 마지막 다섯 번의 반복에서만 warp divergence가 발생하게 된다. 모든 warp divergence가 사라진 것은 아니지만 처음 구현한 `reduceNeighbored`에서 모든 반복 단계에서 warp divergence가 발생하는 것보다는 적다.

이 커널을 추가해 실행시켜보면 아래와 같은 결과를 얻을 수 있다.
```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce          elapsed 21.5972 ms    cpu sum: 2139353471
gpu Neighbored      elapsed 0.5586 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess  elapsed 0.3065 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
```

필자의 환경에서 약 1.8배의 속도 향상을 얻을 수 있다.

두 커널의 차이점은 `nsight compute`를 통해 `inst_per_warp`를 측정해보면 알 수 있다. `inst_per_warp`는 warp 당 실행한 명령어의 수를 측정한 결과이다.
```
$ sudo ncu --metrics smsp__average_inst_executed_per_warp.ratio ./reduce_integer
```
위의 커맨드로 프로파일링한 결과는 다음과 같다.

```
reduceNeighbored(int *, int *, unsigned int), 2023-Feb-12 18:37:15, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         292.88
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceNeighboredLess(int *, int *, unsigned int), 2023-Feb-12 18:37:15, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         115.38
  ---------------------------------------------------------------------- --------------- ------------------------------
```

Warp divergence를 개선한 버전의 `reduceNeighborLess` 커널이 warp 당 실행한 명령어의 수가 2배 이상 적다는 것을 알 수 있다. 즉, 더 적은 명령어로 동일한 작업을 수행했다는 것을 의미한다.

두 커널의 차이는 global memory load throughput을 측정을 통해서도 알아볼 수 있다. 다음의 커맨드를 실햏하면,
```
$ sudo ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second ./reduce_integer
```

아래와 같은 출력 결과를 얻을 수 있다.
```
reduceNeighbored(int *, int *, unsigned int), 2023-Feb-12 18:39:14, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         840.80
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceNeighboredLess(int *, int *, unsigned int), 2023-Feb-12 18:39:14, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Tbyte/second                           1.55
  ---------------------------------------------------------------------- --------------- ------------------------------
```

`reduceNeighboredLess` 커널이 `reduceNeighbored` 커널보다 약 2개 가량 더 높은 처리량을 보여준다. 즉, 메모리 액세스 패턴을 통해 메모리 load/store 효율을 증가시켜, 결과적으로 성능을 향상시킬 수 있다는 것을 암시한다.

<br>

# Reducing with Interleaved Pairs

이번에는 interleaved pair 방식으로 구현한 parallel reduction 커널을 살펴본다. Interleaved pair 방식의 커널에서는 neighbored pair와 달리 `stride` 값을 스레드 블록 크기의 반으로 시작하여 매 반복마다 절반이 된다. 각 스레드들은 현재 `stride` 값만큼 떨어진 두 요소의 부분합을 계산하게 된다. 이를 시각적으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpdn51%2FbtrY2onc38d%2FbHMMYu1rdz4TGcE9RpKnqk%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

이 방식에서 reduction을 수행하는 스레드는 변하지 않는다. 단지 각 스레드에서 global memory에 load/store하는 위치만 다를 뿐이다.

Interleaved 방식의 parallel reduction 커널 구현은 다음과 같다.
```c++
__global__
void reduceInterleaved(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            in[tid] += in[tid + stride];
        
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

앞서 언급했듯, 이 커널에서 두 요소 간 stride 값은 스레드 블록 크기의 절반으로 시작된다. 그리고 각 반복마다 절반으로 감소한다.
```c++
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { ... }
```

그리고 `if (tid < stride)` 조건은 첫 번째 반복에서 스레드 블록의 처음 절반의 스레드들만 reduction 연산을 수행하도록 하며, 두 번째 반복에서는 스레드 블록의 처음 1/4의 스레드들만 reduction 연산을 수행하도록 한다.

커널을 추가하여 컴파일한 뒤, 실행하면 다음의 출력 결과를 얻을 수 있다.
```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce              elapsed 21.5847 ms    cpu sum: 2139353471
gpu Neighbored          elapsed 0.5591 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess      elapsed 0.3075 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceInterleaved   elapsed 0.2660 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
```

처음 구현한 `reduceNeighbored` 커널보다 `reduceInterleaved` 커널이 약 2.1배 가량 빠르다는 것을 보여준다.

`nsight compute`를 통해 global memory load throughput을 측정해보면 다음과 같이 출력된다.
```
reduceNeighbored(int *, int *, unsigned int), 2023-Feb-12 18:58:46, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         842.23
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceNeighboredLess(int *, int *, unsigned int), 2023-Feb-12 18:58:46, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Tbyte/second                           1.55
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceInterleaved(int *, int *, unsigned int), 2023-Feb-12 18:58:46, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         466.71
  ---------------------------------------------------------------------- --------------- ------------------------------
```

기존의 두 커널 `reduceNeighbored`와 `reduceNeighboredLess`보다 더 낮은 처리량을 보여준다. 수치적으로는 더 적은 처리량을 보여주지만 이는 처리 성능이 낮아졌다는 것이 아닌, 더 적은 메모리 액세스로 동일한 작업을 수행했다고 해석할 수 있다.

`reduceInterleaved` 커널애서 발생하는 warp divergence는 `reduceNeighboredLess` 커널에서 발생하는 것과 동일한데, 이는 `branch efficiency`를 측정해보면 알 수 있다. Branch efficiency는 아래 커맨드의 metrics로 측정할 수 있다.
```
$ sudo ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./reduce_integer
```

출력 결과는 다음과 같다.
```
reduceNeighboredLess(int *, int *, unsigned int), 2023-Feb-12 19:03:07, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                          98.36
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceInterleaved(int *, int *, unsigned int), 2023-Feb-12 19:03:07, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                          98.36
  ---------------------------------------------------------------------- --------------- ------------------------------
```

두 커널에서 모두 branch efficiency가 98.36%로 측정된다.


<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher