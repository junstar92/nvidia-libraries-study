# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introducing the CUDA Memory Model](#introducing-the-cuda-memory-model)
  - [Benefits of a Memory Hierarchy](#benefits-of-a-memory-hierarchy)
- [CUDA Memory Model](#cuda-memory-model)
  - [Registers](#registers)
  - [Local Memory](#local-memory)
  - [Shared Memory](#shared-memory)
  - [Constant Memory](#constant-memory)
  - [Texture Memory](#texture-memory)
  - [Global Memory](#global-memory)
- [GPU Caches](#gpu-caches)
- [CUDA Variable Declaration Summary](#cuda-variable-declaration-summary)
- [Static Global Memory](#static-global-memory)
- [References](#references)

# Introducing the CUDA Memory Model

모든 프로그래밍 언어에서 메모리 액세스와 관리는 중요하다. 특히 메모리 관리는 가속기에서 성능에 아주 큰 영향을 미친다.

대부분의 워크로드는 얼마나 빠르게 데이터를 불러오고 저장하는지에 제한되어 있기 때문에 low-latency와 high-bandwidth memory가 성능에 유리하다. 고용량, 고성능 메모리가 항상 주어지는 것이 아니므로 주어진 하드웨어 메모리 시스템에서 최적의 latency와 bandwidth를 달성하려면 메모리 모델에 달려있다. CUDA 메모리 모델은 별도의 host memory와 device memory 시스템을 통합하고, 전체 메모리 계층을 사용자에게 노출시킨다. 따라서, 최적의 성능을 달성하기 위해 데이터의 배치를 명시적으로 제어할 수 있다.

<br>

## Benefits of a Memory Hierarchy

일반적으로 프로그램에서는 어떤 시점에서 임의의 데이터에 액세스하거나 임의의 코드를 실행할 수 없다. 대신, **locality 원칙** 을 따른다. 이 원칙은 특정 시점에서 address space의 작고 지역화된 부분에 액세스할 것을 제안한다. Locality에는 두 가지 타입이 있다.

- Temporal locality (locality in time)
- Spartial locality (locality in space)

**Temporal locality**는 만약 어떤 데이터가 참조되었을 때, 짧은 시간 내에 이 데이터는 다시 참조될 가능성이 높고 시간이 지날수록 참조될 가능성이 낮다는 것을 의미한다. **Spartial locality**는 어떤 메모리 위치가 참조되었다면, 이 메모리 근처 또한 참조될 가능성이 높다는 것을 의미한다.

요즘 컴퓨터들은 점진적으로 latency는 낮아지지만 capacity는 큰 메모리 계층을 사용하여 성능을 최적화한다. 이러한 메모리 계층은 locality 원칙 때문에 유용하다. 메모리 계층 구조는 다양한 latency, bandwidth, capacity를 갖는 여러 메모리들로 구성된다. 일반적으로 processor-to-memory latency가 증가하면 메모리 capacity도 증가하며, 전형적인 메모리 계층은 아래 그림과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcEyGqZ%2FbtrZoqmp0gu%2FNKGY1b1zZRn7xTA1GpWn91%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

아래쪽에 위치하는 메모리들은 일반적으로 다음의 특징들을 가지고 있다.

- Lower cost per bit
- Higher capacity
- Higher latency
- Less frequently accessed by the processor

CPU와 GPU의 main memory는 DRAM이지만, CPU L1 캐시와 같은 lower-latency memory는 SRAM이다. 위와 같은 메모리 계층 구조에서 데이터는 프로세서에서 계속해서 사용중일 때는 low-latency, low-capacity 메모리에서 저장되어 있고, 사용대기 중인 데이터들은 high-latency, high-capacity 메모리에 저장된다.

GPU와 CPU는 둘다 메모리 계층 설계에서 비슷한 원칙과 모델을 사용한다. 주요 차이점은 CUDA 프로그래밍 모델에서 메모리 계층 구조가 더 많이 노출되어 있고 더 많은 제어가 가능하다는 것이다.

<br>

# CUDA Memory Model

개발자 관점에서 일반적으로 메모리를 다음의 두 분류로 나눌 수 있다.

- **Programmable** : 어떤 데이터를 programmable memory에 위치하도록 명시적으로 제어할 수 있다
- **Non-programmable** : 데이터의 위치를 제어할 수 없고, automatic technique을 통해 좋은 성능을 기대한다

CPU 메모리 계층 구조에서 L1/L2 캐시가 non-programmable memory에 해당한다. 반면 CUDA memory model에서는 아래와 같은 다양한 programmable memory를 제공한다.

- Registers
- Shared memory
- Local memory
- Constant memory
- Texture memory
- Global memory

아래 그림은 CUDA의 메모리 계층 구조를 보여준다. 각각의 메모리는 서로 다른 scope, lifetime, caching behavior을 갖는다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdFF9qB%2FbtrZsEXK0Hj%2FB8IwbA5GTL3899XXTlMxk1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

커널 내 스레드는 own private local memory를 가진다. 스레드 블록은 own shared memory를 가지며, 이 메모리는 같은 스레드 블록 내의 모든 스레드들에게 visible이다. 그리고 shared memory의 내용은 스레드 블록의 lifetime동안 지속된다. 그리고, 모든 스레드들은 global memory에 액세스할 수 있다. 모든 스레드들이 액세스할 수 있는 두 개의 읽기전용 메모리 공간이 있는데, 하나는 constant memory이고 다른 하나는 texture memory이다.

Global, constant, texture memory는 서로 다른 용도에 최적화되어 있다. Texture memory의 경우, 다양한 address mode와 다양한 data layout을 필터링하는 기능을 제공한다. Global, constant, texture memory는 application과 동일한 lifetime을 가진다.

## Registers

**Registers**는 GPU에서 가장 빠른 memory space 이다. 커널 내에서 어떠한 한정자(ex, `__shared__`)도 없이 선언되는 automatic 변수는 일반적으로 register에 저장된다. 커널에서 선언되는 배열 또한 register에 저장되지만, 컴파일 시간에 배열의 크기가 결정되는 경우에만 register에 저장된다.

Register 변수는 각 스레드에 private이다. 일반적으로 커널 내에서 자주 액세스되는 thread-private 변수에 사용된다. 이 변수는 커널과 lifetime이 같으며 커널 수행이 완료되면 액세스할 수 없다.

Register는 SM의 active warp 간에 분배되는 매우 희소한 리소스이다. Ampere 아키텍처의 경우, 한 스레드에서 최대 255개의 register를 사용할 수 있다. 하지만 device query를 해보면([Device Query](/cuda/study/04_device_query.md) 참조), RTX3080의 경우에 한 블록에서 가능한 최대 register의 수는 65536개인데, 블록 당 최대 스레드 갯수가 1024개이다. 따라서, 스레드 블록에 1024개의 스레드가 있다면, 결과적으로 한 스레드에서 사용가능한 register의 수는 `65535 / 1024 = 64`개가 된다.

커널에서 register를 덜 사용한다면 더 많은 스레드 블록이 SM에 상주할 수 있다. 즉, 더 많은 스레드 블록이 동시에 active될 수 있다는 것을 의미하고 occupancy와 성능이 향상된다.

> 구현한 커널이 사용하는 하드웨어 리소스는 `nvcc`의 `--resource-usgae` 옵션을 추가해주면 확인해볼 수 있다. 이와 관련해서는 간단하게 [Understanding Warp Execution](/cuda/study/06_understanding_warp_execution.md#occupancy)에서 간단히 언급한 적이 있으므로 이를 참조 바람

만약 커널에서 제한된 것보다 더 많은 register를 사용한다면, 초과된 register들은 local memory로 넘어가게 된다. 이를 **register spilling** 이라고 부르며, 이는 성능 하락에 큰 영향을 미친다. `nvcc` 컴파일러는 휴리스틱을 사용하여 register 사용량을 최소화하고 register spilling을 피할 수 있다. 휴리스틱을 사용하려면 [`__launch_bound__()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds) 한정자를 사용하여 컴파일러에게 필요한 부가적인 정보를 제공하면 된다.

```c++
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
MyKernel(...)
{
    ...
}
```

Launch bound로 지정된 것보다 더 많은 스레드 또는 블록으로 커널을 실행하려고 하면, launch fail이 발생하게 된다.

또한, 컴파일러를 통해 모든 커널에 대해서 사용할 수 있는 register의 최대 갯수를 지정할 수 있는데, `-maxrregcount=<N>` 옵션을 사용한다. 단, launch bound가 지정된 커널 함수에서는 `maxrregcount`가 무시된다.

## Local Memory

**Local memory**는 오직 automatic 변수로 선언될 때 액세스할 수 있는 메모리이다. [Register](#registers)에서 언급했듯이 커널 내에서 automatic으로 선언되는 변수는 일반적으로 register에 저장되지만, 아래의 조건들을 만족하는 automatic 변수들은 local memory에 저장된다.

- 컴파일 시간에 결정되지 않는 인덱스 값으로 참조되는 배열(local array)
- Register space에 저장하기에 너무 큰 구조체 또는 배열
- 커널에서 사용 가능한 register를 넘어서는 모든 변수들 (`register spilling`이 발생한 경우에 해당됨)

`Local Memory`라는 이름은 사실상 잘못 명명되었다고 볼 수 있는데, local memory에 저장되는 값들은 사실 물리적으로 global memory(device memory)와 동일한 위치이다. 따라서, local memory에 액세스하는 것은 global memory와 동일하게 high latency, low bandwidth의 특징을 가지고 있다.

단, local memory는 연속적인 스레드 ID가 연속된 32bit words에 액세스하도록 구성되므로, 워프 내 모든 스레드가 동일한 상대 주소에 액세스한다면 이 액세스는 완전히 병합되게 된다 (예를 들어, 배열에서 동일한 인덱스에 액세스 또는 구조체 변수의 동일한 멤버에 액세스하는 등).

## Shared Memory

커널 내에서 `__shared__` 속성으로 선언되는 변수는 **shared memory**에 저장된다.

Shared memory는 on-chip이므로 global 또는 local memory보다 높은 bandwidth와 낮은 latency라는 특징을 가지고 있다. CPU의 L1 cache와 유사하지만, shared memory는 programmable 하다.

각 SM(streaming processor)은 스레드 블록 간에 분배되는 제한된 크기의 shared memory를 가지고 있다. 따라서, 이 메모리는 커널 내에서 과도하게 사용되지 않도록 주의해야 한다. 만약 너무 많이 사용하게 되면 SM에서 active warp의 수가 제한된다.

Shared memory는 커널 함수 scope에서 선언되지만 스레드 블록과 lifetime을 공유한다. 스레드 블록의 실행이 종료될 때 shared memory는 해제되고, 다른 스레드 블록에 할당된다.

Shared memory는 스레드 간 통신을 위한 기본 수단이 될 수 있다. 한 블록 내의 스레드들은 shared memory에 저장된 데이터를 공유하는 방식으로 서로 협력할 수 있다. Shared memory애 액세스할 때는 반드시 아래의 CUDA Runtime API를 호출하여 동기화해주는 것이 필요하다.

```c++
void __synchthreads();
```

이 함수는 같은 스레드 블록 내의 모든 스레드들이 이 함수 호출에 도달해야 다음 단계를 수행할 수 있도록 해주는 배리어를 생성한다. 이를 통해 잠재적인 **race condition**를 방지할 수 있다. 서로 다른 스레드에서 동일한 메모리 위치에 액세스할 때, 그 순서는 정의되지 않는다. 동일한 메모리 위치에 액세스할 때, 이러한 다중 액세스 중 하나가 write 라면 문제가 발생할 수 있다. 다만, `__syncthreads()` 호출은 SM을 빈번하게 idle 상태로 만들기 때문에 성능에 영향을 미칠 수 있다는 점에 유의해야 한다.

## Constant Memory

**Constant memory**는 device memory에 상주하며 각 SM에서 전용 constant cache에 캐싱된다. 상수 변수는 `__constant__` 속성으로 선언된다.

상수 변수는 커널 외부에서 global scope로 선언되어야 한다. 모든 compute capability에서 64KB로 제한되어 있으며 정적으로 선언되어 동일한 컴파일 유닛의 모든 커널에서 visible 이다.

커널 함수 내에서는 constant memory는 오직 read-only이다. 따라서, constant memory는 커널 내에서 사용하기 전에 host 측에서 아래 CUDA API를 통해 초기화를 해주어야 한다.

```c++
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
```

Constant memory는 warp 내의 모든 스레드들이 동일한 메모리 주소를 읽을 때 최상의 성능을 얻을 수 있다. 예를 들어, 어떤 수학 공식에서 계수가 바로 constant memory를 사용하는 좋은 예시가 될 수 있다 (warp 내의 모든 스레드들은 이 공식으로 계산할 때 입력 데이터가 다르더라도 계수는 동일한 값을 사용하므로).

## Texture Memory

**Texture memory** 또한 device memory에 상주하며, 각 SM마다 read-only cache에 캐싱된다. On-chip에 캐시되므로 global memory보다 효율적인 bandwidth를 달성할 수 있으며, Cache miss가 발생할 때만 device memory로부터 read하는 cost가 발생한다.

이 메모리는 액세스 패턴이 **spartial locality**가 상당한 경우를 위해 설계되었는데, spartial locality는 인접한 스레드들이 인접한 메모리 주소에 액세스할 가능성이 높은 경우를 의미한다.

> 주로 그래픽을 처리하는 경우에 자주 사용되는 것으로 보인다

Texture cache는 2D spartial locality에 최적화되어 있으며, warp 내 스레드들이 texture memory를 사용해 2D data에 액세스할 때 최상의 성능을 얻을 수 있을 것으로 확인된다. 특히 이미지 처리를 할 때 filtering 등의 작업을 하드웨어에서 처리할 수 있어서 성능 상의 이점을 얻을 수 있다. 하지만 대부분의 어플리케이션에서는 global memory 보다 느리다.

## Global Memory

**Global memory**는 GPU에서 가장 용량이 크고, 가장 큰 latency를 가지는 메모리이다. 일반적으로 가장 흔하게 사용되는 메모리이다. 여기서 `global`이라는 이름은 이 메모리의 scope와 lifetime을 지칭한다. 이 메모리는 해당 GPU device의 어떠한 SM에서도 액세스할 수 있으며, lifetime은 어플리케이션의 lifetime과 동일하다.

Global memory에 저장되는 변수는 정적, 동적으로 선언할 수 있다. Device cod에서 정적으로 global memory를 선언하려면 `__device__`를 사용하여 변수를 선언하면 된다.

동적으로 global memory를 할당하려면 `cudaMalloc()`을 사용하고, 이렇게 할당된 메모리는 `cudaFree()`를 사용하여 해제한다.

> 이 CUDA API를 사용하는 방법은 [CUDA Programming Model](/cuda/study/02_cuda_programming_model.md#managing-memory)에서 간단히 살펴봤었으니 필요하면 참조 바람

이렇게 동적으로 할당된 global memory는 커널 함수에 포인터로 전달한 뒤, 커널 함수 내에서 사용할 수 있다. 할당된 global memory는 어플리케이션의 lifetime 동안 지속되며 모든 커널의 모든 스레드에서 액세스할 수 있다. 모든 스레드에서 global memory에 액세스할 수 있으며, 스레드 블록 간의 스레드들은 서로 동기화될 수 없으므로 global memory를 사용할 때는 race condition에 주의해야 한다. 다른 스레드 블록의 스레드들이 동일한 global memory 위치를 수정하면 undefined behavior이 발생하게 된다.

Global memory는 device memory에 상주하며, 32-bytes, 64-bytes, 또는 128-bytes memory transactions로 액세스된다. 최적의 성능을 얻으려면 memory transaction을 최적화하는 것이 필수적이다. 한 warp에서 memory load/store를 수행할 때, 필요한 memory transaction의 수는 일반적으로 아래 두 가지 요인에 따라 달라진다.

- Distribution of memory addresses across the threads of that warp
- Alignment of memory addresses per transaction

최근 GPU device에서 global memory access에 대한 requierment가 비교적 완화되었는데, 이는 memory transaction이 캐싱되기 때문이다. 이렇게 캐싱된 memory transaction는 data locality를 활용하여 throughput efficiency를 향상시킨다.

> Global memory에서의 성능은 compute capability마다 다르므로, 정확한 내용은 각 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 문서를 참조해야 한다.

> [Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)에서 global memory access 패턴에 따라서 성능에 어떻게 영향을 미치는지 살펴볼 수 있다.

<br>

# GPU Caches

CPU cache와 마찬가지로 GPU caches는 non-programmable memory 이다. GPU device에는 다음의 4가지 타입의 cache가 있다.

  - L1
  - L2
  - Read-only constant
  - Read-only texture

SM마다 각자 하나의 L1 cache가 있으며, 모든 SM들이 공유하는 L2 cache가 있다. L1와 L2는 local/global memory에 데이터를 저장하는데 사용된다 (register spill 포함).

CPU에서는 memory load와 store 모두 캐싱될 수 있다. 하지만 GPU에서는 오직 memory load operation만 캐싱될 수 있으며, memory store operation은 캐싱될 수 없다.

각 SM은 하나의 read-only constant cache와 read-only texture cache가 존재한다.

<br>

# CUDA Variable Declaration Summary

아래 표는 어떻게 CUDA 변수를 선언하느냐에 따라 해당 변수가 어떤 메모리에 위치하는지, scope/lifespan은 어떻게 되는지를 보여준다.

|Qualifier|Variable Name|Memory|Scope|Lifespan|
|---------|-------------|------|-----|--------|
||`float var`|Register|Thread|Thread|
||`float var[100]`|Local|Thread|Thread|
|`__device__`|float var|Global|Global|Application|
|`__constant__`|float var|Constant|Global|Application|
|`__shared__`|float var|Shared|Block|Block|

아래는 각 메모리 타입의 특징을 요약한 표이다.

|Memory|On/Off Chip|Cached|Access|Scope|Lifetime|
|--|--|--|--|--|--|
|Register|On-chip|n/a|R/W|1 thread|Thread|
|Local|Off-chip|Yes*|R/W|1 thread|Thread|
|Shared|On-chip|n/a|R/W|All threads in block|Block|
|Global|Off-chip|Yes*|R/W|All threads + host|Host allocation|
|Constant|Off-chip|Yes|R|All threads + host|Host allocation|
|Texture|Off-chip|Yes|R|All threads + host|Host allocation|

> **\*** Global memory의 경우, compute capability 6.0, 7.x에서는 기본적으로 L1과 L2에 캐싱된다. 이보다 낮은 compute capability에서는 기본적으로 L2에서만 캐싱되며, 컴파일 플래그를 통해 L1에 캐싱되도록 할 수도 있다.

> **\*** Local memory는 compute capability 5.x 이외의 device에서는 L1과 L2에 캐싱되며, compute capability 5.x device에서는 오직 L2에만 캐싱된다.

<br>

# Static Global Memory

개인적으로 static global memory를 사용한 적이 거의 없는데, 아래 코드를 통해서 간단히 어떻게 static global memory를 사용할 수 있는지 살펴볼 수 있다.

```c++
#include <stdio.h>
#include <cuda_runtime.h>

__device__ float dev_data;

__global__
void checkGlobalVariable()
{
  // display the original value
  printf("Device: the value of the global variable is %f\n", dev_data);
  // after the value
  dev_data += 2.0f;
}

int main(int argc, char** argv)
{
  // initialize the static global variable
  float value = 3.14f;
  cudaMemcpyToSymbol(dev_data, &value, sizeof(float));
  printf("Host:   copied %f to the static global variable\n", value);
  
  // launch the kernel
  checkGlobalVariable<<<1,1>>>();
  
  // copy the static global variable back to the host
  cudaMemcpyFromSymbol(&value, dev_data, sizeof(float));
  printf("Host:   the value changed by the kernel to %f\n", value);
  
  cudaDeviceReset();
  return 0;
}
```

위 코드의 실행 결과는 다음과 같다.
```
Host:   copied 3.140000 to the static global variable
Device: the value of the global variable is 3.140000
Host:   the value changed by the kernel to 5.140000
```

Host code에서는 device 변수에 직접 액세스할 수 없고, device code 또한 host 변수에 직접 액세스할 수 없다. 따라서, 아래의 함수를 사용해서 device global 변수에 액세스한다.

```c++
cudaMemcpyToSymbol(dev_data, &value, sizeof(float));
```

위 API의 첫 번째 인자는 참조자로 받기 때문에 `&` 를 붙여주지 않아도 된다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)