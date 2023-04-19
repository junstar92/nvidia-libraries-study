# Table of Contents

- [Table of Contents](#table-of-contents)
- [Maximize Memory Throughput](#maximize-memory-throughput)
  - [Data Transfer between Host and Device](#data-transfer-between-host-and-device)
  - [Device Memory Accesses](#device-memory-accesses)
- [References](#references)

<br>

# Maximize Memory Throughput

전체적인 메모리 처리량을 극대화하기 위한 첫 번째 단계는 낮은 bandwidth로 수행되는 data transfers를 최소화하는 것이다.

이는 host와 device 간의 data transfer를 최소화하는 것을 의미하며, 아래의 [Data Transfer between Host and Device](#data-transfer-between-host-and-device)에서 조금 더 설명한다. Host와 device 간 data transfer의 bandwidth는 global memory와 device 간의 data transfer보다 훨씬 낮다.

메모리 처리량을 극대화하는 것은 on-chip memory(shared memory 및 L1/L2, texture, constant cache)를 최대한으로 사용하여 global memory와 device 간의 data transfer를 최소화하는 것을 의미하기도 한다.

Shared memory는 `user-managed cache`와 동일하다. 즉, 일반적인 캐시는 사용자가 직접 컨트롤할 수 없지만, GPU의 shared memory는 사용자가 명시적으로 할당하고 액세스할 수 있다.

특정 어플리케이션의 경우, shared memory가 아닌 기존의 hardware-managed cache를 사용하는 것이 더 적합할 수 있다. Compute capability 7.x 이상의 device에서는 동일한 on-chip memory가 L1 cache와 shared memory에 사용된다. 즉, **L1 cache와 shared memory는 동일한 on-chip memory에 상주한다는 것을 의미**한다. L1 캐시와 shared memory에 할당되는 메모리 양은 어플리케이션 내에서 설정 가능하다.

커널 함수의 메모리 액세스 처리량은 각 메모리 타입에 대한 액세스 패턴에 따라 크게 달라질 수 있다. 따라서, 메모리 처리량을 최대화하려면 [Device Memory Accesses](#device-memory-accesses)에서 설명하고 있는 각 메모리의 최적의 액세스 패턴을 기반으로 가능한 한 최적의 액세스 패턴을 구성하는 것이다. 특히 global memory는 on-chip memory에 비해 memory bandwidth가 매우 낮기 때문에 최적화하지 않으면 성능에 큰 영향을 미치게 된다.

## Data Transfer between Host and Device

어플리케이션에서는 host와 device 간의 data transfer를 최소화해야 한다. 이를 달성하기 위한 한 가지 방법은 host 코드를 device로 옮기는 것이다. 이는 왠만하면 데이터 처리에 관한 코드들은 device에서 처리되도록 한다는 것을 의미한다. 심지어 병렬 처리로 성능 향상이 크기 않다거나 병렬로 구현하기 애매한 커널이라도 왠만해선 host로 다시 데이터를 복사하는 것보다 더 좋을 수 있다는 것이다.

또한, data transfer 연산에는 오버헤드가 크기 때문에 각 transfer를 개별적으로 수행하는 것보다 하나의 큰 transfer로 처리하는 것이 항상 성능이 더 좋다.

**Front-side bus**가 있는 시스템인 경우, page-locked host memory를 사용하여 host와 device 간의 data transfer 성능을 높일 수 있다.

**Mapped page-locked memory**를 사용하는 경우, device memory를 할당하거나 device<->host 간의 명시적인 데이터 복사가 필요없다. 이 메모리에 대한 transfer는 커널이 이 메모리에 액세스할 때마다 암시적으로 수행된다. Mapped page-locked memory는 단 한 번만 읽거나 쓴다고 가정한다면, device와 host 간의 명시적인 데이터 복사보다 성능이 더 좋을 수 있다.

> Device memory와 host memory가 물리적으로 통합되어 있는 시스템에서는 host와 device 간의 복사가 불필요하므로 mapped page-locked memory를 사용해야 한다. 이는 `device query`를 통해 해당 시스템의 메모리가 통합되어 있는지 확인할 수 있다.

## Device Memory Accesses

아래 포스팅들은 각 메모리의 특성과 각 메모리에 대한 최적의 액세스 패턴에 대해 설명하고 있으니, 필요하면 참조 바란다 (아래에 설명하는 내용들을 커버).

- [Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)
  - [Matrix Transpose Problem](/cuda/study/11-1_matrix_transpose_problem.md)
- [Shared Memory](/cuda/study/12_shared_memory.md)
  - [Data Layout of Shared Memory](/cuda/study/12-1_data_layout_of_shared_memory.md)
  - [Reducing Global Memory Access](/cuda/study/12-2_reducing_global_memory_access.md)
  - [Coalescing Global Memory Accesses](/cuda/study/12-3_coalescing_global_memory_accesses.md)


Addressable memory(i.e., global, local, shared, constant, or texture memory)에 액세스하는 instruction은 warp 내 스레드들 간의 메모리 주소 분포에 따라 여러 번 re-issue될 수 있다. 스레드 간 주소 분포가 instruction throughput에 미치는 영향은 메모리 타입에 따라서 다르며, 이는 아래에서 자세히 설명한다.

### Global Memory

Global memory는 device memory에 상주하며, 32-, 64-, 또는, 128-byte memory transactions로 액세스 된다. 이 memory transactions는 반드시 정렬되어 있어야 하며, 오직 정렬된 32-, 64-, or, 128-byte segments만 memory transactions으로 읽거나 쓸 수 있다 (첫 번째 주소가 32, 64, or, 128의 배수인 메모리).

한 warp가 global memory에 액세스하는 instruction을 실행할 때, 이 warp 내 스레드들의 메모리 액세스는 각 스레드가 액세스하는 word의 크기와 스레드들 간의 메모리 액세스 분포에 따라서 한 개 이상의 memory transaction으로 병합된다. 일반적으로 더 많은 transaction이 필요할수록 사용되지 않는 word가 전달되기 때문에 instruction throughput은 감소한다.

> [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x), [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x) 그리고 [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-0)에서 각 compute capabilities에 대해 global memory 액세스를 처리하는 방법을 자세히 언급하고 있다.

Global memory throughput을 극대화하기 위해서는 메모리 액세스를 병합(`coalesed access`)하는 것이 중요하다.

> Global memory access 패턴에 대한 자세한 내용은 [Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)를 참조 바람

### Size and Alignment Requirement

Global memory instruction은 1,2,4,8,or,16 바이트 크기의 word에 대한 read/write를 지원한다. Global memory에 상주하는 데이터에 대한 모든 액세스(by variable or pointer)는 오직 데이터 타입의 크기가 1,2,4,8,or16 바이트이면서 정렬되어 있어야만 하나의 global memory instruction으로 컴파일된다.

만약 데이터 타입의 크기나 정렬 요구사항을 충족하지 못한다면, 이러한 액세스는 interleaved access patterns을 사용하여 여러 instruction로 컴파일된다. 따라서, global memory의 데이터에 대해서 `size / alignment requirement`를 만족하도록 하는 것이 좋다.

> [Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types)들은 자동으로 alignment requirement를 만족시킨다.

구조체의 경우에는 아래와 같이 alignment specifiers인 `__align__(8)`, `__align__(16)`을 사용하여 컴파일러에 의해서 크기와 정렬 요구사항을 강제할 수 있다.
```c++
struct __align__(8) {
    float x;
    float y;
};
```
or
```c++
struct __align__(16) {
    float x;
    float y;
    float z;
};
```

Driver/Runtime API에 의해 반환되는 변수의 주소는 항상 적어도 256 바이트로 정렬된다.

> 정렬되지 않은 8바이트 또는 16바이트 word를 읽으면 잘못된 결과를 발생시킬 수 있으므로 유의해야 한다.

### Two-Dimensional Arrays

일반적으로 global memory 액세스 패턴에서 각 스레드의 인덱스 `(tx, ty)`를 사용할 때, 2차원 배열의 한 요소에 액세스하는 주소는 아래와 같이 계산할 수 있다. 여기서 2차원 배열의 `width`와 해당 배열의 첫 요소의 시작 주소인 `BaseAddress`가 사용된다.
```
BaseAddress + width * ty + tx
```
이러한 액세스가 완전히 병합(coalesced)되려면 스레드 블록의 width와 배열의 `width`가 모드 warp 크기의 배수가 되어야 한다.

특히, width가 warp 크기의 배수가 아닌 경우, 배열의 각 행에 padding을 추가하여 warp 크기의 배수로 맞추어 주는 것이 훨씬 더 효율적으로 액세스할 수 있다는 것을 의미한다. API 중 `cudaMallocPitch()`나 `cuMemAllocPitch()`를 사용하여 하드웨어에 종속되지 않고 이러한 제약 조건들을 만족하는 배열을 할당할 수 있다.

### Local Memory

Local memory 액세스는 특정 `automatic variables`에서 발생하는 액세스이다. 컴파일러에 의해서 local memory에 위치할 수 있는 automatic variable의 조건은 다음과 같다.

- 상수로 인덱싱되지 않는 배열
- 너무 많은 레지스터를 사용하는 큰 구조체나 배열
- 커널에서 사용할 수 있는 레지스터를 초과해서 사용하게 되는 모든 변수 (known as `register spilling`)

컴파일할 때 `-ptx`와 `-keep` 옵션을 추가하여 PTX 어셈블리 코드를 살펴보면 변수가 local memory 영역에 위치하는지 살펴볼 수 있다. Local memory 영역에 위치하는 변수는 첫 번째 컴파일 단계에서 `.local` mnemonic을 사용하여 선언되고, `ld.local`과 `st.local` mnemonics를 사용하여 액세스된다.

첫 번째 컴파일 단계에서 지정되지 않더라도, 타겟 아키텍처에서 너무 많은 레지스터 공간을 사용하는 것으로 판단하면 후속 컴파일 단계에서 local memory 영역으로 배치할 수 있다. 이는 `cuobjdump`를 사용하여 cubin object를 확인하면 알 수 있다. 또한, 컴파일할 때 `--ptxas-options=-v` 옵션을 지정하면, 컴파일러는 커널 당 total local memory usage(`lmem`)를 리포트한다.

Local memory는 device memory에 상주하기 때문에 local memory access는 global memory access와 똑같이 높은 latency와 낮은 bandwidth를 가지며, 동일한 메모리 병합 요구조건을 가진다.

> Compute capability 5.x 이상의 device에서 local memory access는 global memory access와 같은 방식으로 항상 L2에 캐싱된다.

### Shared Memory

Shared memory는 on-chip memory이므로, 높은 bandwidth와 global(or local) memory보다 훨씬 낮은 latency가 특징이다.

높은 memory bandwidth를 달성하기 위해서 shared memory는 동일한 크기의 메모리 모듈로 나뉘는데, 이 메모리 모듈을 뱅크(bank)라고 부른다. 뱅크에 대한 액세스는 동시에 수행될 수 있다. 따라서, 서로 다른 n개의 메모리 뱅크에 속하는 n개의 주소로 구성된 모든 메모리 read 또는 write는 동시에 처리될 수 있으므로 전체 bandwidth는 하나의 모듈의 bandwidth보다 n배 높다.

그러나 만약 요청된 메모리 중 두 주소가 하나의 동일한 메모리 뱅크에 속한다면, `bank conflict`가 발생되고 이 액세스는 순차적으로 수행된다. 하드웨어는 bank conflict가 있는 메모리 요청을 conflict가 발생하지 않도록 하는 별도의 메모리 요청으로 분할한다. 결과적으로 분할되는만큼 처리량이 감소된다. 별도로 분할되는 메모리 요청 수가 n개라면, 초기 메모리 요청에서 **n-way** bank conflict가 발생했다고 말한다.

성능을 극대화하려면 bank conflict를 최소화해야 하며, 따라서, 메모리 주소가 어떻게 메모리 뱅크에 매핑되는지 이해하는 것이 중요하다.

> Shared memory에 대한 자세한 내용은 아래 포스트를 참조 바람
> - [Shared Memory](/cuda/study/12_shared_memory.md)

> 공식 문서의 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)에는 각 compute capability에서 shared memory에 대한 내용을 다룬다.

### Constant Memory

Constant memory 또한 device memory에 상주한다. 다만, constant cache에 의해 캐싱될 수 있다.

초기 메모리 요청에서 서로 다른 메모리 주소에 대한 요청만큼 개별적은 메모리 요청으로 분할되어 처리된다. 따라서 그만큼 처리량이 감소된다. 최종적으로 캐시 히트인 경우에는 constant cache의 처리량으로 처리되고, 그렇지 않다면 device memory의 처리량으로 처리된다.

> Constant memory에 대해서 아래 포스트에서 다루고 있다.
> - [Constant Memory](/cuda/study/13_constant_memory.md)

### Texture and Surface Memory

Texture and surface memory는 device memory에 상주하며, texture cache에 의해 캐싱된다. 따라서, texture fetch나 surface read에서는 cache miss인 경우에만 딱 한 번 device memory로부터의 read cost가 필요하다. Texture cache는 2D spatial locality에 최적화되어 있으므로, 2D에서 서로 근접하는 texture 또는 surface 주소를 읽는 동일한 warp 내 스레드에서 최상의 성능을 달성할 수 있다.

Texture fetch나 surface read를 사용하면 device memory read보다 몇 가지 이점들이 존재하는데, 이는 [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)에서 확인할 수 있다.

<br>

# References

- [NVIDIA CUDA Documentations: Maximize Memory Throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput)