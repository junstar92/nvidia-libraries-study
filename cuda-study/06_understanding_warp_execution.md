# Table of Contents

- [Table of Contents](#table-of-contents)
- [Understanding Warp Execution](#understanding-warp-execution)
  - [Warps and Thread Blocks](#warps-and-thread-blocks)
- [Warp Divergence](#warp-divergence)
- [Resource Partitioning within a Warp](#resource-partitioning-within-a-warp)
- [Occupancy](#occupancy)
  - [Checking Active Warps with Nsight Compute](#checking-active-warps-with-nsight-compute)
- [References](#references)

<br>

# Understanding Warp Execution

커널을 실행(launch)할 때, software 관점에서는 모든 스레드가 동시에 실행되는 것처럼 보인다. 논리적 관점(logical view)에서 이 말은 사실이다. 하지만 하드웨어 관점(hardware view)에서는 모든 스레드가 물리적으로 동시에 병렬로 실행되지 않는다. [CUDA Execution Model](/cuda-study/05_cuda_execution_model.md)에서 언급했듯이 CUDA는 32개의 스레드를 하나의 execution unit으로 그룹화하여 태스크를 수행하게 된다. 이번 포스팅에서는 하드웨어 관점에서의 warp execution에 대해 조금 더 자세히 살펴본다.

## Warps and Thread Blocks

`Warps`는 SM에서 기본 실행 단위(execution unit)이다. 스레드 블록들로 구성된 하나의 그리드(커널)를 실행할 때, 그리드 내 스레드 블록들은 SMs로 분배된다. SM에 스레드 블록이 스케쥴링되면, 스레드 블록 내의 스레드들은 warps로 분할된다. 하나의 warp는 연속된 32개의 스레드들로 구성되며, warp 내 모든 스레드들은 SIMT 방식으로 실행된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlLlKt%2FbtrYTqsTuB1%2FtMXOtsYuP0asGIE1g79eGK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

스레드 블록은 1,2,3차원으로 구성될 수 있다. 하지만, 하드웨어 관점에서 살펴보면 모든 스레드는 1차원으로 정렬된다. 각 스레드는 블록 내에서 unique ID를 가진다. 1차원 스레드 블록에서 unique thread ID는 CUDA 내장 변수인 `threadIdx.x`에 저장되고, 연속된 `threadIdx.x` 값을 갖는 32개의 스레드가 하나의 warp로 그룹화된다. 예를 들어, 128개의 스레드로 구성된 1차원 스레드 블록은 아래와 같이 4개의 warps로 조직된다.

```
Warp 0: thread  0, thread  1, thread  2, ..., thread 31
Warp 1: thread 32, thread 33, thread 33, ..., thread 63
Warp 2: thread 64, thread 65, thread 66, ..., thread 95
Warp 3: thread 96, thread 97, thread 98, ..., thread 127
```

2차원 또는 3차원 스레드 블록의 논리적 레이아웃은 `x`, `y`, `z`차원을 이용하여 1차원의 물리적 레이아웃으로 변환될 수 있다. 예를 들어, 2차원 스레드 블록에서 각 스레드에 대한 ID는 내장 변수 `threadIdx`와 `blockDim` 변수를 통해 아래와 같이 계산된다.

```
threadIdx.x + blockDim.x * threadIdx.y
```

비슷하게 3차원 스레드 블록에서는 아래와 같이 계산된다.

```
threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z
```

스레드 블록에서의 warp 수는 아래와 같이 계산된다.

$$ \text{Warps per Block} = \text{ceil}\left(\frac{\text{Threads per Block}}{\text{Warp Size}}\right) $$

하드웨어에서는 항상 warp를 discrete number로 할당한다. 따라서, 만약 스레드 블록의 크기가 warp의 크기로 나누어 떨어지지 않는다면, 마지막 warp에서 몇몇 스레드들은 비활성으로 남게된다.

만약 40x2 크기의 2차원 스레드 블록(80개의 스레드)이 있을 때, 하드웨어에서는 이 스레드 블록을 3개의 warp를 할당하게 된다. 결과적으로 80개의 스레드를 위해서 96개의 하드웨어 스레드를 구성하게 되며, 마지막 warp에서 절반의 스레드들은 inactive 상태가 된다. 이때, 비록 inactive 스레드들이 사용되지 않더라도 이 스레드들에 대한 SM의 리소스는 소비하게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQbTwr%2FbtrYXkd8syX%2FJUwCJHn7qfz1ZrkDa0HVa0%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

> Logical 관점에서 스레드 블록은 1D, 2D, or 3D layout으로 구성된다.
>
> 하지만, hardware 관점에서 스레드 블록은 1D의 warps로 구성된다. 스레드 블록의 스레드들은 1D layout으로 구성되며, 연속된 32개의 스레드들은 하나의 warp를 형성한다.

<br>

# Warp Divergence

CUDA 프로그래밍에서는 전통적인 C 스타일의 flow-control 구조, 즉, `if ... else`, `for`, `while`을 지원한다.

CPU는 branch prediction을 위해 복잡한 하드웨어로 구성되며 각 조건 체크마다 어떤 브랜치를 취하는지 예측한다. 만약 예측이 맞다면, branching은 약간의 성능 페널티만을 발생시킨다. 하지만 예측이 올바르지 않다면 CPU는 몇 cycle 동안 스톨(stall)될 수 있다. 여기서 CPU가 복잡한 control flow를 어떻게 잘 처리하는지 완벽히 이해할 필요는 없다.

GPU는 복잡한 branch prediction 메커니즘이 없는 비교적 단순한 장치이다. Warp 내의 모든 스레드들은 반드시 동일한 cycle에서 동일한 instruction을 실행한다. 만약 한 스레드가 어떤 instruction을 실행하면, warp 내의 모든 스레드들은 반드시 이 instruction을 수행하게 된다. 그런데 같은 warp 내의 스레드들이 다른 branch 경로를 취하게 되면 문제가 발생한다. 예를 들어, 아래의 구문을 스레드들이 수행한다고 가정해보자.

```
if (cond) {
    ...
}
else {
    ...
}
```

Warp 내 16개의 스레드들이 이 코드에서 `cond`가 `true`이고, 나머지 16개의 스레드들은 `false` 이라고 한다면, 16개의 스레드들은 `if` 블럭 내의 코드를 수행하지만 나머지 16개의 스레드들은 `else` 블럭 내의 코드를 수행해야 한다. 같은 warp 내의 스레드들이 다른 instruction을 수행하는 것을 **warp divergence** 라고 칭한다. Warp divergence는 우리가 알고 있는 '한 warp 내의 모든 스레드들은 각 cycle에서 동일한 instruction을 수행한다'라는 사실에 모순을 일으키는 것처럼 보인다.

Warp divergence가 발생하면, warp는 각 branch path를 순차적으로 실행한다. 이때, 하나의 branch path를 실행할 때, 이 branch path를 수행하지 않는 스레드들은 비활성화시킨다. 따라서, warp divergence는 상당한 성능 저하를 일으킨다. 위에서 본 예시에서는 `if` branch path를 수행할 때 오직 16개의 스레드들만 활성화되고, 나머지는 비활성화되어 병렬 처리 능력이 절반이 된다. 만약 더 많은 conditional branch가 있다면 병렬 처리 효과는 더욱 저하된다.

> **branch divergence** 는 오직 warp 내에서만 발생한다. Warp들간에 발생하는 branching은 warp divergence를 일으키지 않는다. (예를 들어, warp 0의 모든 스레드들은 `if` 블럭 코드를 수행하고, warp 1의 모든 스레드들은 `else` 블럭 코드를 수행하는 것)

아래 그림은 warp divergence가 발생했을 때, 스레드들이 어떻게 수행되는지를 보여준다. 조건이 `true`인 스레드들이 `if` 블럭의 코드를 수행할 때, `false`인 스레드들은 `true`인 스레드들의 실행이 완료될 때까지 스톨된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbDIKbX%2FbtrYSjnzAoH%2Fe1NBBuDotM3LN9IQN3F3w1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

최상의 성능을 달성하려면, 동일한 warp 내에서의 다른 execution path를 피해야 한다. 스레드 블록 내에서 스레드의 warp 할당은 결정적(deterministic)이라는 것에 주목한다면, 같은 warp 내의 모든 스레드들이 같은 control path를 취하도록 데이터를 파티셔닝할 수 있다.

간단한 커널 함수들을 통해서 warp divergence에 대해 살펴보자.

> 전체 코드는 [warp_divergence.cu](/code/cuda/warp_divergence/warp_divergence.cu)를 참조 바람.

먼저, 아래의 코드처럼 커널이 2개의 branch를 가지도록 작성한다. 이 코드에서는 데이터를 짝수/홀수 스레드로 파티셔닝하는데, warp divergence를 일으키기 때문에 상당히 좋지 않은 코드이다. `(tid % 2 == 0)`이라는 조건은 짝수 ID의 스레드들은 `if`문을 수행하고 홀수 ID의 스레드들은 `else`문을 수행하도록 한다.

```c++
__global__
void mathKernel1(float* c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0.f, b = 0.f;

    if (tid % 2 == 0) {
        a = 100.f;
    }
    else {
        b = 200.f;
    }
    c[tid] = a + b;
}
```

반면에 아래 코드는 interleave approach로 데이터를 파티셔닝하여 warp divergence를 피한다. 따라서, GPU를 100% 활용하게 된다. `(tid/warpSize) % 2 == 0` 조건은 warp 단위로 warp 내의 모든 스레드들이 `true`또는 `false`가 되도록 한다. 위의 커널과 결과는 같지만, 이 코드가 성능이 더 좋다.
```c++
__global__
void mathKernel2(float* c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0.f, b = 0.f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.f;
    }
    else {
        b = 200.f;
    }
    c[tid] = a + b;
}
```

[warp_divergence.cu](/code/cuda/warp_divergence/warp_divergence.cu)를 아래의 커맨드를 통해 컴파일하고,

```
$ nvcc -o warp_divergence warp_divergence
```

`warp_divergence`를 실행하면 아래와 같은 출력을 얻을 수 있다.
```
> Data size : 64
Execution Configure (block 64, grid 1)
warmingup  <<<    1   64 >>> elapsed time: 0.021376 msec 
mathKernel1<<<    1   64 >>> elapsed time: 0.003136 msec 
mathKernel2<<<    1   64 >>> elapsed time: 0.003072 msec
```

> 여기서 커널의 성능을 자세히 평가하기 위해서 warmingup 커널을 먼저 실행시켜서 GPU를 warm-up 시켜준다. warmingup 커널이 없으면, 첫 번째로 실행되는 커널의 성능은 제대로 측정되지 않는다.

> 코드만 봤을 때, `mathKernel1`은 `mathKernel2`에 비해 병렬 처리 능력이 2배정도 떨어진다고 볼 수 있지만, 실제 수행 시간은 그렇지 않다. 이 결과는 `nvcc` 컴파일러가 알아서 최적화해주는 부분이 적용되어 있기 때문이다. 이 부분은 바로 아래에서 `nsight compute`를 통해 확인할 수 있다.
>
> 하지만, 최적화 옵션을 끄기 위해서 `-g -G` 옵션을 추가하여 컴파일하더라도 실제 속도 측면에서 성능 차이는 크게 발생하지 않는 것을 볼 수 있다. 개인적으로 GPU(RTX3080)의 성능이 좋고, `nvcc`(V11.8)로 컴파일할 때 최적화 옵션을 끄더라도 어느정도 컴파일러가 최적화를 수행하는 것으로 추측된다.

NVIDIA에서 제공하는 커널 프로파일러인 `nsight compute`를 사용하면 실제로 warp divergence가 얼마나 발생하는지 볼 수 있다. 먼저, 아래 커맨드를 통해서 branch efficiency를 측정해볼 수 있다.

```
$ sudo ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./warp_divergence
```

필자가 수행했을 때 결과는 아래와 같다.

```
mathKernel1(float *), 2023-Feb-11 22:29:47, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                              0
  ---------------------------------------------------------------------- --------------- ------------------------------

mathKernel2(float *), 2023-Feb-11 22:29:47, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                              0
  ---------------------------------------------------------------------- --------------- ------------------------------
```

놀랍게도 branch efficiency가 0%로 측정된다. 이와 같은 결과는 컴파일할 때, `nvcc`에 의해서 최적화가 적용되었기 때문이다. 최적화로 인해서 branch가 모조리 없어졌음을 아래 커맨드를 통해서 확인할 수 있다.

```
$ sudo ncu --metrics smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum ./warp_divergence
```

위 커맨드를 통해서 각 커널의 branch 수와 divergence가 발생한 branch의 수를 측정할 수 있다. 그 결과는 아래와 같이 출력되었다.

```
mathKernel1(float *), 2023-Feb-11 22:32:12, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_branch_targets.sum                                                                                       0
  smsp__sass_branch_targets_threads_divergent.sum                                                                     0
  ---------------------------------------------------------------------- --------------- ------------------------------

mathKernel2(float *), 2023-Feb-11 22:32:12, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_branch_targets.sum                                                                                       0
  smsp__sass_branch_targets_threads_divergent.sum                                                                     0
  ---------------------------------------------------------------------- --------------- ------------------------------
```

즉, 코드 상에서는 분명 branch가 존재하지만, 최적화로 인해서 모든 branch가 없어졌다는 것을 확인할 수 있다. 이러한 최적화를 비활성화하려면 컴파일할 때 `-G` 옵션을 주면 된다. 모든 최적화를 비활성화하는 것은 아닌 것으로 보이는데, 일단 대부분의 최적화는 비활성화되는 것으로 보인다.
```
$ nvcc -o warp_divergence -G warp_divergence.cu
```

위의 커맨드로 컴파일한 뒤, 다시 `nsight compute`로 branch efficiency를 확인해보면 아래와 같이 출력되는 것을 확인할 수 있다.
```
mathKernel1(float *), 2023-Feb-11 22:51:53, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                             80
  ---------------------------------------------------------------------- --------------- ------------------------------

mathKernel2(float *), 2023-Feb-11 22:51:53, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_average_branch_targets_threads_uniform.pct                                %                            100
  ---------------------------------------------------------------------- --------------- ------------------------------
```

Branch efficiency는 `(total branches - divergent branches) / total branches`로 계산할 수 있다. 이 값들은 
```
$ sudo ncu --metrics smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum ./warp_divergence
```
위 커맨드를 통해 구할 수 있고, 출력은 다음과 같다.
```
mathKernel1(float *), 2023-Feb-11 22:53:57, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_branch_targets.sum                                                                                      10
  smsp__sass_branch_targets_threads_divergent.sum                                                                     2
  ---------------------------------------------------------------------- --------------- ------------------------------

mathKernel2(float *), 2023-Feb-11 22:53:57, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  smsp__sass_branch_targets.sum                                                                                       9
  smsp__sass_branch_targets_threads_divergent.sum                                                                     0
  ---------------------------------------------------------------------- --------------- ------------------------------
```

따라서, `mathKernel1`의 branch efficiency는 `(10-2)/10 = 0.8 (80%)`로 계산되고, `mathKernel2`의 branch efficiency는 `(9-0)/9 = 1.0 (100%)`로 계산되는 것을 `nsight compute`를 통해 확인할 수 있다.

> `nsight compute`를 통해 출력되는 branch의 수가 조금 이상하고, `mathKernel1`의 코드만 봤을 때 branch efficiency가 50%가 되어야 할 것 같지만 80%로 측정되고 있다. 컴파일러의 `-G` 옵션으로는 모든 최적화가 비활성화되는 것이 아닐 수도 있다고 생각되어 [Completely disable optimizations on NVCC](https://stackoverflow.com/questions/11821605/completely-disable-optimizations-on-nvcc)에서 언급하는 최적화 비활성화 옵션들을 모두 사용해봤지만, 여전히 `mathKernel1`의 branch efficiency는 80%로 출력되었다.
>
> 참고로 `nsight compute`로 측정되는 branch 수나 branch efficiency는 `nvcc` 컴파일러의 버전에 따라 다를 수 있다. V11.4의 `nvcc`로 컴파일하여 테스트해보니 아래와 같이 출력되는 것으로 확인된다.
> ```
> mathKernel1(float *), 2023-Feb-11 23:00:01, Context 1, Stream 7
>  Section: Command line profiler metrics
>  ---------------------------------------------------------------------- --------------- ------------------------------
>  smsp__sass_branch_targets.sum                                                                                      12
>  smsp__sass_branch_targets_threads_divergent.sum                                                                     2
>  ---------------------------------------------------------------------- --------------- ------------------------------
>
> mathKernel2(float *), 2023-Feb-11 23:00:01, Context 1, Stream 7
>   Section: Command line profiler metrics
>   ---------------------------------------------------------------------- --------------- ------------------------------
>   smsp__sass_branch_targets.sum                                                                                      11
>   smsp__sass_branch_targets_threads_divergent.sum                                                                     0
>   ---------------------------------------------------------------------- --------------- ------------------------------
> ```

<br>

# Resource Partitioning within a Warp

Warp의 local execution context는 주로 아래의 리소스들로 구성된다.

- Program counters
- Registers
- Shared memory

RTX3080 정보를 runteim API로 쿼리하면 아래와 같은 정보를 얻을 수 있다 ([Device Qeury](/cuda-study/04_device_query.md) 참조).
```
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total shared memory per multiprocessor:        102400 bytes
Total number of registers available per block: 65536
```

각 SM은 register file에 저장된 32-bit register들의 집합을 가지고 있는데, 이 register들은 스레드 간에 파티셔닝된다. RTX3080에서는 각 SM 당 64K 32-bit registers를 가지고 있다. 또한, shared memory는 스레드 블록 간에 파티셔닝된다. RTX3080에서 각 SM은 100KB의 shared memory를 가지고 있으며, 블록 당 최대 48KB의 shared memory가 배분된다.

따라서, 주어진 커널에 대해 SM에서 사용 가능한 register의 수와 shared memory 크기와 이 커널에서 필요한 register의 수와 shared memory 크기에 따라서 SM에 동시에 상주할 수 있는 스레드 블록 및 warp의 수가 달라진다.

아래 그림은 각 스레드가 더 많은 register를 소모할 때, 더 적은 warp들이 SM에 위치할 수 있다는 것을 보여준다. 만약 커널 함수가 사용하는 register의 수를 줄일 수 있다면, 더 많은 warp가 SM에 상주할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZiGx8%2FbtrYRVAyIsg%2FykHZSkuKgPrmLqrCQ1fYnK%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

아래 그림은 스레드 블록들이 더 많은 shared memory를 소모할 때, 더 적은 스레드 블록들이 SM에서 동시에 수행된다는 것을 보여준다. 마찬가지로 각 스레드 블록이 사용하는 shared memory의 크기를 줄일 수 있다면, 더 많은 스레드 블록이 동시에 수행될 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbEaoCS%2FbtrYSj84gOs%2FM0bhx9SYvoVVpcK0rIvuG1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

이처럼 이용 가능한 리소스에 따라 SM에 상주하는 스레드 블록의 수가 제한된다. 각 SM당 register의 수와 shared memory의 크기는 GPU마다 다르다. 따라서 어떤 GPU에서는 커널 함수가 실행되지만, 리소스가 적은 GPU에서는 커널 함수 실행이 실패할 수 있다.

Register나 shared memory와 같은 리소스가 할당된 스레드 블록을 **active block** 이라고 부른다. Active block이 포함하는 warp는 **active warps** 라고 부른다. Active warps는 크게 아래 3가지 타입으로 분류할 수 있다.

- Selected warp (실행중인 warp)
- Stalled warp (아직 실행할 준비가 되지 않은 warp)
- Eligible warp (실행할 준비는 되었지만 현재 실행중이 아닌 warp)

SM에 상주하는 Warp는 아래 두 가지 조건이 만족되면 eligible warp가 된다.
- 32 CUDA Cores are available for execution
- All arguments to the current instruction are read 

<br>

# Occupancy

각 CUDA 코어 내에서 instruction은 순차적으로 수행된다. SM 내에서 하나의 warp가 스톨(stall)되면, SM은 다른 eligible warp를 실행하도록 switching 한다. 이상적으로 CUDA 코어들이 occupied한 상태에 있도록 하려면 SM 내에는 충분한 수의 warp가 필요하다. **Occupancy** 는 SM 당 maximum warps와 active warps의 비율이다. 

$$ \text{occupancy} = \frac{\text{active warps}}{\text{maximum warps}} $$

여기서 SM당 maximum warps의 수는 `cudaGetDeviceProperties(...)` runtime API를 통해 구할 수 있다. [device_query.cpp](/code/cuda/device_query/device_query.cpp) 코드에서 SM당 가능한 최대 warp의 수를 계산하여 출력하는데, 아래와 같이 계산한다. RTX3080의 경우에는 SM당 최대 48개의 warp가 가능하다.
```
printf("  Maximum number of warps per multiprocessors:      %d\n", dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize);
```

CUDA Toolkit을 다운받으면 `CUDA_Occupancy_Calculator.xls`라는 엑셀 파일을 제공한다. 이 파일에서는 특정 GPU에서 실행할 커널의 occupancy를 극대화하기 위한 그리드와 커널의 사이즈를 계산해준다. 이를 계산하기 위해서는 아래의 커널 리소스 정보를 입력해주어야 한다.

- Threads per block (execution configuration)
- Registers per thread (resource usage)
- Shared memory per block (resource usage)

참고로 커널에서의 register와 shared memory 사용량은 `nvcc`로 컴파일할 때, `--resource-usage` 옵션을 추가하면 얻을 수 있다. 예를 들어, 아래와 같이 커맨드를 입력하면,
```
$ nvcc -o warp_divergence --resource-usage warp_divergence.cu
```

다음과 같은 출력 결과를 얻을 수 있다 ([Printing Code Generation Statistics](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics) 참조).

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z11mathKernel2Pf' for 'sm_86'
ptxas info    : Function properties for _Z11mathKernel2Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 360 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11mathKernel1Pf' for 'sm_86'
ptxas info    : Function properties for _Z11mathKernel1Pf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 360 bytes cmem[0]
ptxas info    : Compiling entry function '_Z9warmingupPf' for 'sm_86'
ptxas info    : Function properties for _Z9warmingupPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 360 bytes cmem[0]
```

각 커널에 대한 리소스 사용량이 출력되며, 여기서 출력되는 정보는 다음과 같다. 이 정보는 NVIDIA NVCC 문서와 stackoverflow의 내용을 참조했다.
- `registers` : register file on every SM
- `gmem` : global memory
- `smem` : shared memory
- `cmem[N]` : constant memory bank with index N
  - `cmem[0]` : bank reserved for kernel argument and statically-size constant values
  - `cmem[2]` : user defined constant objects
  - `cmem[16]` : compiler generated constants
- `stack frame` : per thread stack usage by a function
- `spill stores/loads` : stores and loads doen on stack memory (when it couldn't be allocated to physical registers)

## Checking Active Warps with Nsight Compute

Nsight Compute를 통해 커널의 occupancy 비율을 측정할 수 있다. [matrix_add2.cu](/code/cuda/matrix_add/matrix_add2.cu) 코드를 사용하여, 블록 사이즈에 따라 occupancy가 어떻게 변하는지 확인해보자. 이 코드에서는 간단히 행렬 덧셈을 2D block approach로 구현한 커널을 테스트한다.

```c++
__global__
void sumMatrixOnGPU2D(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}
```

[matrix_add2.cu](/code/cuda/matrix_add/matrix_add2.cu)를 컴파일하고, 스레드 블록의 사이즈를 (32,32), (32,16), (16,32), (16,16)으로 각각 지정하여 실행시킨 결과는 다음과 같다.

```
$ ./matrix_add 32 32
> Matrix size: 16384 x 16384
> sumMatrixOnGPU2D<<<(512,512), (32,32)>>> (Average)Elapsted Time: 4.706 msec

$ ./matrix_add 32 16
> Matrix size: 16384 x 16384
> sumMatrixOnGPU2D<<<(512,1024), (32,16)>>> (Average)Elapsted Time: 4.623 msec

$ ./matrix_add 16 32
> Matrix size: 16384 x 16384
> sumMatrixOnGPU2D<<<(1024,512), (16,32)>>> (Average)Elapsted Time: 4.633 msec

$ ./matrix_add 16 16
> Matrix size: 16384 x 16384
> sumMatrixOnGPU2D<<<(1024,1024), (16,16)>>> (Average)Elapsted Time: 4.621 msec
```

사실 GPU(RTX3080) 성능이 꽤 좋은 편이라 속도 측면에서 눈에 띄는 차이점이 발견되지는 않는다. 하지만 아래 커맨드를 통해 nsight compute로 `achieved_occupancy`를 측정하면 스레드 블록 크기에 따라 어떤 차이가 있는지 관찰할 수 있다. **achieved occupancy**는 SM에서 지원하는 maximum warp 수 대비 cycle 당 평균 active warp의 비율을 의미한다. `<bx>`와 `<by>`에 각각 블록의 x,y 차원의 크기를 입력해준다.

```
sudo ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./matrix_add <bx> <by>
```

스레드 블록 크기 (32,32), (32,16), (16,32), (16,16) 순으로 출력 결과는 아래와 같다.
```
sumMatrixOnGPU2D(const float *, const float *, float *, int, int), 2023-Feb-12 00:20:22, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          48.07
  ---------------------------------------------------------------------- --------------- ------------------------------

sumMatrixOnGPU2D(const float *, const float *, float *, int, int), 2023-Feb-12 00:20:29, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          57.38
  ---------------------------------------------------------------------- --------------- ------------------------------

sumMatrixOnGPU2D(const float *, const float *, float *, int, int), 2023-Feb-12 00:20:35, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          64.28
  ---------------------------------------------------------------------- --------------- ------------------------------

sumMatrixOnGPU2D(const float *, const float *, float *, int, int), 2023-Feb-12 00:20:42, Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------------- --------------- ------------------------------
  sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          67.61
  ---------------------------------------------------------------------- --------------- ------------------------------
```

테스트한 스레드 블록 크기 순서대로 48.07, 57.38, 64.28, 67.61이라는 결과를 얻었다. 이 결과를 통해 아래의 결론을 얻을 수 있다.

- 4번째 케이스가 가장 높은 achieved occupancy를 보여주지만, 이 커널이 다른 커널에 비해 빠른 것은 아니다. 따라서, occupancy가 높다고 더 좋은 성능을 보여주는 것은 아니며, 성능에 영향을 주는 다른 요인이 있다고 볼 수 있다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- (stackoverflow) [Completely disable optimizations on NVCC](https://stackoverflow.com/questions/11821605/completely-disable-optimizations-on-nvcc)
- (stackoverflow) [CUDA constant memory banks](https://stackoverflow.com/questions/12290708/cuda-constant-memory-banks)