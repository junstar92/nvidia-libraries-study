# Table of Contents

- [Table of Contents](#table-of-contents)
- [Unrolling Loops](#unrolling-loops)
- [Reducing with Unrolling](#reducing-with-unrolling)
- [Reducing with Unrolled Warps](#reducing-with-unrolled-warps)
- [Reducing with Complete Unrolling](#reducing-with-complete-unrolling)
- [Load/Store Efficiency of All Kernels](#loadstore-efficiency-of-all-kernels)
- [References](#references)

<br>

# Unrolling Loops

**Loop unrolling** 은 loop maintenance instruction과 branch frequency를 감소시켜 loop execution을 최적화하는 기법이다.

예를 들면, 아래와 같은 코드를

```c++
for (int i = 0; i < 100; i++) {
    a[i] = b[i] + c[i];
}
```
아래와 같이 loop의 수를 절반으로 감소시키고, loop body 내에서 처리하는 코드를 두 번 작성하는 것이다.
```c++
for (int i = 0; i < 100; i += 2) {
    a[i] = b[i] + c[i];
    a[i+1] = b[i+1] + c[i+1];
}
```

여기서 loop body의 copy 수를 **loop unrolling factor** 라고 부른다. 이 값에 따라서 원래 반복 횟수가 얼마나 감소되는지 결정된다. Loop unrolling은 루프를 실행하기 전에 반복 횟수를 알고 있는 경우, 순차 배열 처리 루프의 성능을 향상시키는데 가장 효과적이다.

High-level 코드만 봤을 때, loop unrolling이 어떻게 성능을 향상시키는지 쉽게 알 수 없다. 이러한 성능 향상은 컴파일러가 수행하는 low-level instruction의 최적화에서 비롯된다. 예를 들어, 위의 loop unrolling이 적용된 `for`문에서 `i < 100` 이라는 조건은 50번만 체크된다. 또한, 매 루프에서 수행되는 read/write는 서로 독립적이기 때문에 CPU에서 memory operation은 동시에 실행될 수 있다.

CUDA에서의 loop unrolling은 여러 가지를 의미할 수 있는데, 목적은 성능 향상으로 동일하다. 즉, instruction overhead를 줄이고 더 많은 독립적인 instruction을 스케쥴링하는 것이다. 결과적으로 더 많은 concurrent operation들이 파이프라인에 추가되어 instruction과 memory의 bandwidth를 더 많이 활용한다.

이번 포스팅에서는 [Avoiding Branch Divergence](/cuda-study/07_avoiding_branch_divergence.md)에서 살펴본 reduction problem 문제에 loop unrolling을 적용하여 성능이 얼마나 좋아지는지 확인한다.

> 전체 코드는 [reduce_integer.cu](/code/cuda/reduce_integer/reduce_integer.cu)를 참조바람

<br>

# Reducing with Unrolling

Interleaved 방법으로 구현한 `reduceInterleaved` 커널에서는 하나의 스레드 블록(a thread block)이 스레드 블록 크기 만큼의 요소(a data block)를 처리한다. 만약, 스레드 블록의 크기가 512라면 한 스레드 블록은 512개의 요소를 처리하게 된다. 즉, 하나의 스레드는 하나의 요소를 처리한다는 것을 의미한다.

이 커널에 unrolling을 적용하여 각 스레드 블록에서 하나의 스레드가 두 개의 요소를 처리하도록 할 수 있다. 즉, 하나의 스레드 블록이 자신의 크기의 2배(two data block)를 처리하게 된다. Loop unrolling factor를 2로 지정하여 unrolling 기법을 적용한 `reduceUnrolling2` 커널은 다음과 같이 구현할 수 있다.

```c++
__global__
void reduceUnrolling2(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 2;
    
    // unrolling 2 data blocks
    if (idx + blockDim.x < n) {
        g_in[idx] += g_in[idx + blockDim.x];
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            in[tid] += in[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

위 커널 함수에서는 먼저 이웃한 data block을 더해준다.
```c++
if (idx + blockDim.x < n)
    g_in[idx] += g_in[idx + blockDim.x];
```

배열의 인덱싱 또한 조정되어야 하는데, 하나의 스레드 블록의 두 개의 data block을 처리하므로 아래와 같이 `blockDim.x * 2`로 인덱싱하게 된다.
```c++
unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
// convert global data pointer to the local pointer of this block
int* in = g_in + blockDim.x * blockIdx.x * 2;
```

아래 그림은 각 스레드가 액세스하는 데이터를 보여준다. 기존 커널에서는 각 스레드는 하나의 데이터에만 액세스했지만, `reduceUnrolling2` 커널에서는 각 스레드가 2개의 데이터에 액세스하게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbweYfJ%2FbtrYTpOJYzc%2FMALyJsC23tiO2eLtAaG2NK%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

각 스레드 블록은 자신의 스레드 갯수의 두 배인 데이터를 처리하므로, 커널을 호출할 때 필요한 그리드 크기를 계산하는 방법도 조정이 되어야 한다. 스레드 블록 크기에 대해 실제로 필요한 그리드의 크기는 이전의 커널 대비 절반이므로 계산한 그리드 크기의 절반으로 지정해주어야 한다. 이외에도 커널의 결과가 저장된 `d_out`의 크기도 `grid.x / 2`라는 것에 유의한다.
```c++
reduceUnrolling2<<<grid.x / 2, block>>>(d_in, d_out, num_elements);
```

`reduceUnrolling2`가 추가된 코드를 컴파일하고 실행하면 아래와 같은 출력 결과를 얻을 수 있다.
```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce              elapsed 21.4282 ms    cpu sum: 2139353471
gpu Neighbored          elapsed 0.5626 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess      elapsed 0.3060 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceInterleaved   elapsed 0.2650 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceUnrolling2    elapsed 0.1624 ms     gpu sum: 2139353471 <<<grid 16384 block 512>>>
```

[Avoiding Branch Divergence](/cuda-study/07_avoiding_branch_divergence.md)에서 살펴본 baseline인 `reduceNeighbored` 커널보다 약 3.5배 정도 속도가 향상되었고, `reduceInterleaved` 커널보다는 약 1.6배 정도 속도가 향상되었다.

마찬가지로 unrolling factor가 4, 8인 경우도 간단히 구현할 수 있는데, 각각의 factor로 구현한 커널은 다음과 같다.
```c++
// unrolling 4
__global__
void reduceUnrolling4(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 4;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 3 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            in[tid] += in[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}

// unrolling 8
__global__
void reduceUnrolling8(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 8;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 7 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        sum += g_in[idx + blockDim.x * 4];
        sum += g_in[idx + blockDim.x * 5];
        sum += g_in[idx + blockDim.x * 6];
        sum += g_in[idx + blockDim.x * 7];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            in[tid] += in[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

하나의 스레드 블록이 처리하는 데이터 블록 갯수가 달라졌기 때문에, 커널을 호출하거나 결과를 host로 복사할 때 크기를 재조정해주어야 한다는 점에 유의한다. 위의 두 커널 함수를 적용한 코드를 컴파일하고, 실행하면 아래의 결과를 얻을 수 있다.
```
gpu reduceUnrolling2    elapsed 0.1626 ms     gpu sum: 2139353471 <<<grid 16384 block 512>>>
gpu reduceUnrolling4    elapsed 0.1258 ms     gpu sum: 2139353471 <<<grid 8192 block 512>>>
gpu reduceUnrolling8    elapsed 0.1157 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
```

Loop unrolling을 통해 하나의 스레드에서 더 많은 memory read/load operation 수행하여 memory letency를 hiding하는 효과를 기대할 수 있다. 이러한 성능은 `nsight compute`를 통해 device memory read throughput metric을 측정하여 확인할 수 있다 (unrolling를 적용하면 각 스레드에서 read op가 늘어나므로). 아래의 커맨드를 입력하면 각 커널의 dram read throughput을 얻을 수 있다.
```
$ sudo ncu --metrics dram__bytes_read.sum.per_second ./reduce_integer
```

아래 결과와 같이 unrolling factor가 커질수록 device memory read throughput이 커지는 것을 확인할 수 있다.
```
reduceUnrolling2(int *, int *, unsigned int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  dram__bytes_read.sum.per_second                                           Gbyte/second                         392.01
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceUnrolling4(int *, int *, unsigned int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  dram__bytes_read.sum.per_second                                           Gbyte/second                         557.34
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceUnrolling8(int *, int *, unsigned int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  dram__bytes_read.sum.per_second                                           Gbyte/second                         614.49
  ---------------------------------------------------------------------- --------------- ------------------------------
```

<br>

# Reducing with Unrolled Warps

`__syncthreads()`는 블록 내 동기화(**intra-block synchronization**)에 사용된다. 앞서 살펴본 reduction 커널들을 살펴보면, 매 라운드마다 모든 스레드가 계산한 부분합 결과를 다음 라운드가 시작되기 전에 global memory에 저장하기 위해 사용된다.

이때, 처리해야할 스레드 블록에서 처리해야할 데이터 수가 32개 이하로 남은 경우를 생각해보자. 즉, 하나의 warp만 실행되는 경우에 해당한다. Warp execution은 SIMT로 동작되기 때문에, 각 instruction에는 암시적인 **intra-warp synchronization**이 존재한다. 여기서 reduction의 마지막 6번의 반복을 unrolling하면, 이러한 동기화를 제거하여 성능을 더 끌어올릴 수 있다.
```c++
if (tid < 32) {
    volatile int *vmem = in;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
}
```

Warp 내의 스레드들은 SIMT 동작이 보장되므로, 각 스레드들은 정확히 동일한 사이클에서 같은 line의 코드를 처리하게 된다. 여기서 주목할 부분은 `volatile` 한정자를 사용한 것이다. `volatile`을 사용하게 되면 컴파일러는 해당 메모리의 값이 다른 스레드에 의해서 언제든지 바뀌거나 사용될 수 있다고 가정한다. 따라서, volatile 변수에 대한 참조는 캐시나 레지스터를 쓰지 않고 직접 메모리를 읽거나 쓰도록 한다. 또한, `volatile`이 없다면 컴파일러 또는 캐시가 이러한 메모리에 대한 read/write를 최적화할 수 있기 때문에 의도치 않게 동작할 수 있으므로 여기서는 반드시 `volatile`을 사용해주어야 한다.

앞서 구현한 `reduceUnrolling8` 커널을 기반으로 unrolling warp 기법을 적용하면 다음과 같이 구현할 수 있다.
```c++
// unrolling warps 8
__global__
void reduceUnrollingWarps8(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 8;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        sum += g_in[idx + blockDim.x * 4];
        sum += g_in[idx + blockDim.x * 5];
        sum += g_in[idx + blockDim.x * 6];
        sum += g_in[idx + blockDim.x * 7];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            in[tid] += in[tid + stride];
        }

        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

Execution configuration이나 global ouput array의 크기는 `reduceUnrolling8` 커널과 동일하다.

위 커널을 추가하여 컴파일 후, 실행하면 아래와 같은 출력 결과를 얻을 수 있다.
```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce                  elapsed 21.8346 ms    cpu sum: 2139353471
gpu Neighbored              elapsed 0.5595 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess          elapsed 0.3077 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceInterleaved       elapsed 0.2648 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceUnrolling2        elapsed 0.1630 ms     gpu sum: 2139353471 <<<grid 16384 block 512>>>
gpu reduceUnrolling4        elapsed 0.1265 ms     gpu sum: 2139353471 <<<grid 8192 block 512>>>
gpu reduceUnrolling8        elapsed 0.1171 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
gpu reduceUnrollingWarps8   elapsed 0.1167 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
```
미세한 차이지만, `reduceUnrolling8`보다 약간 더 빠르게 측정된다. 다만, 측정 오차라고 볼 수 있는 정도라서 유의미한 차이는 아닌 것 같다. Unrolling warp 적용 유무에 따른 차이를 확인하려면 `nsight compute`를 통해 `stall_sync` metric을 측정해보면 알 수 있다. 이 metric을 통해 `reduceUnrollingWarps8` 커널이 `reduceUnrolling8` 커널보다 `__syncthreads()`로 인해 지연되는 warp의 수가 더 적다는 것을 보여준다.

```
$ sudo ncu --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct ./reduce_integer
```

위 커맨드의 출력은 다음과 같다.
```
reduceUnrolling8(int *, int *, unsigned int), 2023-Feb-13 20:54:28, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__warp_issue_stalled_barrier_per_warp_active.pct                                 %                          17.20
  ---------------------------------------------------------------------- --------------- ------------------------------

reduceUnrollingWarps8(int *, int *, unsigned int), 2023-Feb-13 20:54:28, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__warp_issue_stalled_barrier_per_warp_active.pct                                 %                           9.53
  ---------------------------------------------------------------------- --------------- ------------------------------
```

<br>

# Reducing with Complete Unrolling

Compile-time에 루프의 반복 횟수를 알고 있다면, 완벽하게 unrolling할 수 있다. GPU에서 블록 당 최대 스레드 갯수는 고정되어 있고, reduction 커널에서 루프의 반복 횟수는 스레드 블록 차원에 의해 결정되므로, reduction loop를 완전히 unrolling할 수 있다.

> 대부분의 경우, GPU에서 블록 당 최대 스레드 갯수는 대부분 1024이다. 아닌 경우도 있으니 확인은 필요하다.

Complete loop unrolling을 적용하여 구현하면 다음과 같다 (`reduceUnrolling8`을 베이스로 작성).
```c++
__global__
void reduceCompleteUnrollWarps8(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 8;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 7 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        sum += g_in[idx + blockDim.x * 4];
        sum += g_in[idx + blockDim.x * 5];
        sum += g_in[idx + blockDim.x * 6];
        sum += g_in[idx + blockDim.x * 7];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

이 커널에 대한 실행 결과는 다음과 같다.
```
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
> grid 32768  block 512
cpu reduce                      elapsed 21.4272 ms    cpu sum: 2139353471
gpu Neighbored                  elapsed 0.5598 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess              elapsed 0.3054 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceInterleaved           elapsed 0.2635 ms     gpu sum: 2139353471 <<<grid 32768 block 512>>>
gpu reduceUnrolling2            elapsed 0.1622 ms     gpu sum: 2139353471 <<<grid 16384 block 512>>>
gpu reduceUnrolling4            elapsed 0.1273 ms     gpu sum: 2139353471 <<<grid 8192 block 512>>>
gpu reduceUnrolling8            elapsed 0.1166 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
gpu reduceUnrollingWarps8       elapsed 0.1157 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
gpu reduceCompleteUnrollWarps8  elapsed 0.1156 ms     gpu sum: 2139353471 <<<grid 4096 block 512>>>
```

`reduceUnrolling8`이나 `reduceUnrollWarps8`와 비교했을 때, 유의미한 차이를 보여주지는 않으며 사실상 측정 오차에 가깝다. 실제로 여러번 실행시켜보면 조금 더 느리게 측정되는 경우도 발생한다. `stall_sync` metric을 측정해봐도 `reduceUnrollWarps8`보다 좋은 측정 값을 보여주지도 않는 것을 확인했다.

> 위 결과를 봤을 땐, 왠만한 경우 unrolling 8 정도면 충분한 성능 향상 효과를 볼 수 있는 것 같다. Reduction 커널의 경우 그리 복잡하지도 않고 커널 내 오버헤드가 많이 발생하는 것도 아니므로 unrolling warp나 complete warp 기법까지 사용될 정도는 아니라고 생각된다.

<br>

위 코드를 자세히 살펴보면, 스레드 블록의 크기가 1024보다 작다면 필연적으로 불필요한 branch overhead가 발생하게 된다는 것을 알 수 있다. CUDA에서는 device function에 대해 `template`을 지원한다. 따라서 reduction 커널을 템플릿 함수로 작성하여 블록의 크기를 compile-time에 알 수 있다면, `reduceCompleteUnrollWarps8` 커널에서 발생하는 불필요한 오버헤드를 줄일 수 있다.

예를 들면, 다음과 같이 구현할 수 있다.
```c++
template<unsigned int BlockSize>
__global__
void reduceCompleteUnroll(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 8;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 7 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        sum += g_in[idx + blockDim.x * 4];
        sum += g_in[idx + blockDim.x * 5];
        sum += g_in[idx + blockDim.x * 6];
        sum += g_in[idx + blockDim.x * 7];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (BlockSize >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

블록 크기는 프로그램 실행 시 변경될 수 있지만, 커널을 호출할 때 블록의 크기는 컴파일 시간에 결정되어야 한다. 따라서 이 커널을 실행하려면 다음과 같은 방식으로 실행해야 한다.
```c++
switch (block_size) {
    case 1024:
    reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_in, d_out, num_elements);
        break;
    case 512:
    reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_in, d_out, num_elements);
        break;
    case 256:
    reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_in, d_out, num_elements);
        break;
    case 128:
    reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_in, d_out, num_elements);
        break;
    case 64:
    reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_in, d_out, num_elements);
        break;
}
```

<br>

# Load/Store Efficiency of All Kernels

```
$ sudo ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./reduce_integer
```

[Avoiding Branch Divergence](/cuda-study/07_avoiding_branch_divergence.md)과 이번 포스팅에서 구현한 모든 커널에 대해 load/store efficiency를 `nsighe compute`를 통해 측정한 결과를 비교하면 다음과 같다. 실행 커맨드는 위와 같다.

> 전체 코드는 [reduce_integer.cu](/code/cuda/reduce_integer/reduce_integer.cu)를 참조 바람

|Kernel|Time (ms)|Load Efficiency (%)|Store Efficiency(%)|
|:--|--:|--:|--:|
|`reduceNeighbored`|0.5552|25.02|25|
|`reduceNeightboredLess`|0.3075|25.02|25|
|`reduceInterleaved`|0.2651|96.15|95.52|
|`reduceUnrolling2`|0.1644|98.04|97.71|
|`reduceUnrolling4`|0.1267|98.68|97.71|
|`reduceUnrolling8`|0.1166|99.21|97.71|
|`reduceUnrollingWarps8`|0.1171|99.43|99.40|
|`reduceCompleteUnrollWarps8`|0.1171|99.43|99.40|
|`reduceCompleteUnroll` (template)|0.1193|99.43|99.40|

> 대체로 memory load/store 효율이 좋을수록 속도도 빨라지는 것을 확인할 수 있다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher