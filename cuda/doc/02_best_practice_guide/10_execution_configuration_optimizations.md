# Table of Contents

- [Table of Contents](#table-of-contents)
- [Execution Configuration Optimizations](#execution-configuration-optimizations)
- [Occupancy](#occupancy)
  - [Calculating Occupancy](#calculating-occupancy)
- [Hiding Register Dependencies](#hiding-register-dependencies)
- [Thread and Block Heuristics](#thread-and-block-heuristics)
- [Effects of Shared Memory](#effects-of-shared-memory)
- [Concurrent Kernel Execution](#concurrent-kernel-execution)
- [Multiple Contexts](#multiple-contexts)
- [References](#references)

<br>

# Execution Configuration Optimizations

좋은 성능을 얻기 위한 핵심 중 하나는 멀티프로세서를 가능한 쉬지 않도록 유지하는 것이다. 멀티프로세서 간 작업의 균형이 맞지 않으면 최상의 성능을 얻을 수 없다. 따라서, 하드웨어를 최대한 활용하도록 스레드와 블록을 사용하여 어플리케이션을 설계하는 것이 중요하다. 여기서 중요한 개념은 다음 섹션에서 설명하는 occupancy(점유율)이다.

경우에 따라서는 여러 개의 독립적인 커널이 동시에 실행되도록 설계하여 하드웨어 활용도를 개선할 수 있다. 이러한 커널 실행을 **concurrent kernel execution**이라고 부른다.

또 다른 중요한 개념 중 하나는 특정 작업에 할당된 시스템 리소스의 관리이다. 이에 대한 내용은 마지막 섹션에서 다룬다.

<br>

# Occupancy

Thread instructions은 CUDA에서 순차적으로 실행된다. 결과적으로 한 warp가 일시 중지되거나 지연될 때, 다른 warp를 실행시키는 것이 latency를 숨기고 하드웨어를 쉬지 않게 하는 유일한 방법이다. 따라서, 멀티프로세서의 active warp의 수와 관련된 일부 메트릭들은 하드웨어가 얼마나 효율적으로 사용 중인지를 결정하는데 중요하다. 이 메트릭이 바로 점유율(occupancy)이다.

점유율(occupancy)는 최대로 가능한 active warps 수 대비 멀티프로세서 당 active warp 수의 비율이다. 가능한 maximum active warps의 수는 `deviceQuery`를 통해 계산할 수 있다.

높은 점유율을 유지하는 것이 항상 더 좋은 성능을 의미하지는 않는다. 점유율이 늘어도 성능이 향상되지 않는 지점이 있다. 그러나, 낮은 점유율은 항상 memory latency를 숨기는 것을 방해하기 때문에 성능 저하를 초래하게 된다.

CUDA 커널에서 필요한 스레드 당 리소스에 의해서 최대 블록 크기가 제한될 수 있다. 여러 하드웨어나 이후에 출시할 toolkit에서도 호환하기 위해 적어도 하나의 스레드 블록이 SM에서 실행될 수 있도록 `__launch_bounds__(maxThreadsPerBlock)`을 사용하여 커널이 실행될 때의 최대 블록 크기를 지정하는 방법이 있다. 실행에 실패하게 되면 "too many resources requested for launch" 에러가 발생된다. 두 인자를 전달하는 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`를 사용하면 일부 케이스에서 성능을 향상시킬 수 있다. 인자의 값의 커널 분석을 통한 세부 사항으로 결정될 수 있다.

## Calculating Occupancy

점유율을 결정하는 여러 요인 중 하나는 **register availability** 이다. 스레드가 low latency access로 로컬 변수를 유지하기 위해서 레지스터를 사용하는데, 이 레지스터(known as the *register file*)는 멀티프로세서의 모든 스레드들이 공유하는 리소스이므로 제한되어 있다. 레지스터는 전체 블록에 한 번에 할당된다. 따라서, 각 스레드 블록이 너무 많은 레지스터를 사용하면, 멀티프로세서에 상주할 수 있는 스레드 블록의 수가 줄어들어 점유율이 감소한다. 스레드 당 사용할 최대 레지스터 갯수는 `-maxrregcount`를 사용하거나 `__launch_bounds__` qualifier를 사용하여 컴파일 시간에 설정할 수 있다.

점유율을 계산할 때, 각 스레드에서 사용하는 레지스터의 수가 핵심 요소 중 하나이다. 예를 들어, compute capability 7.0의 device에서 각 멀티프로세서는 65,536개의 32비트 레지스터를 가지고 있으며, 최대 2,048개의 스레드가 동시에 상주할 수 있다 (64 warps x 32 threads per warp). 이는 멀티프로세서가 100%의 점유율을 달성하려면 각 스레드가 최대 32개의 레지스터를 사용할 수 있다는 것을 의미한다. 그러나, 이러한 접근 방식은 register allocation granularity를 고려하지는 않는다.

예를 들어, compute capability 7.0인 device에서 스레드 블록이 128개의 스레드로 구성되며 각 스레드 당 37개의 레지스터를 사용하는 커널에서 스레드 블록의 각 warp는 최소 1184개의 레지스터가 필요하다(32 x 37 = 1184). 여기서 하나의 멀티프로세서에는 65536개의 레지스터를 사용할 수 있으므로, 멀티프로세서에 상주할 수 있는 최대 warp의 수는 55개이다. 각 스레드 블록은 128개의 스레드로 구성되어 있기 때문에 각 스레드 블록은 4개의 warp가 있고, 즉, 최대 13개의 스레드 블록이 상주할 수 있다(55/4=13.75). 따라서, 점유율은 13x128/2048=0.8125 이다.

반면, 320개의 스레드로 구성된 스레드 블록이면서, 각 스레드 당 동일하게 37개의 레지스터를 사용하는 경우에는 가능한 최대 warp의 수가 동일한 55개이지만 각 스레드 블록에 스레드가 320개(10 warps)가 있으므로 최대로 상주 가능한 스레드 블록의 수는 5개이다(55/10=5.5). 따라서, 점유율은 5x320/2048=0.78125 이다.

게다가, 레지스터 할당은 warp 당 가장 가까운 256개의 레지스터로 반올림된다.

사용 가능한 레지스터의 수, 각 멀티프로세서에 상주하는 최대 스레드 갯수 및 register allocation granularity는 compute capability마다 다르다. 또한, shraed memory도 스레드 블록 간 분배되기 때문에 레지스터의 사용량과 점유율 사이의 정확한 상관 관계를 결정하기 어려울 수 있다. `nvcc`에서 `--ptxas-options=v` 옵션을 지정하면, 각 커널에서 사용되는 레지스터의 수를 자세히 살펴볼 수 있다. 각 하드웨어에서의 레지스터 할당 공식은 [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading)을 참조하고, 각 compute capability에서의 레지스터 갯수는 [Technical Specifications per Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability) 표를 참조하면 된다.

또한, toolkit에서는 개발자가 다양한 시나리오를 쉽게 테스트할 수 있는 **occupancy calculator**를 Excel 형식으로 제공한다. CUDA Toolkit의 tools 하위 디렉토리에 `CUDA_Occupancy_Calculator.xls`로 저장되어 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/using-cuda-occupancy-calculator-usage.png" width=900px style="display: block; margin: 0 auto; background-color: white"/>

또는, NVIDIA Nsight Compute를 사용하여 점유율을 확인할 수 있다. 이는 Occupancy 섹션에서 확인 가능하다.

어플리케이션에서 CUDA Runtime API를 사용할 수도 있는데, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`을 사용하면 동적으로 launch configuration을 선택할 수 있다.

> Occupancy 관련 Runtime API는 [Maximize Utilization: Occupancy Calculator](/cuda/doc/01_programming_guide/05-02_maximize_utilization.md#occupancy-calculator)에서도 확인할 수 있다.

<br>

# Hiding Register Dependencies

Instruction이 이전 instruction으로부터의 결과가 저장된 레지스터를 사용할 때, register dependency가 발생한다. 대부분의 산술 연산에서 이 latency는 compute capability 7.0 이상의 device에서 일반적으로 4사이클이다. 따라서, 스레드가 이전 명령어에서의 산술 결과를 사용하려면 약 4사이클을 대기해야 한다. 그러나 이러한 시간은 다른 warp의 스레드를 실행하는 것으로 hiding할 수 있다.

> Register Dependecies에 대한 내용은 [Maximize Utilization: Multiprocessor Level](/cuda/doc/01_programming_guide/05-02_maximize_utilization.md#multiprocessor-level)에서 다루고 있다.

<br>

# Thread and Block Heuristics

> 블록 당 스레드의 수는 32의 배수로 하는 것이 좋다. 이를 통해 최적의 computing efficiency을 얻을 수 있고 병합(coalescing)을 활용할 수 있다.

그리드의 블록 사이즈와 차원, 블록 당 스레드의 크기와 차원은 모두 중요한 요소이다. 이러한 다차원의 매개변수는 CUDA에 보다 쉽게 매핑할 수 있도록 하며 성능에는 영향을 미치지 않는다. 따라서, 여기서는 차원보다는 크기에 대해 다룬다.

Latency hiding과 점유율(occupancy)은 리소스(register and shared memory)의 제약과 함께 execution parameters에 의해 암시적으로 결정되는 멀티프로세서 당 active warp의 수에 따라 달라진다. Execution parameters를 선택하는 것은 latency hiding(occupancy)와 resource utilization 간의 균형을 맞추는 문제이다.

각 Execution configuration parameters에는 특정 휴리스틱이 적용된다. 첫 번째 파라미터인 그리드 당 블록 수 또는 그리드 크기를 선택할 때, 고려해야할 사항은 GPU를 바쁘게 유지하는 것이다. 그리드 내 블록 수는 최소한 모든 멀티프로세서가 실행할 블록이 하나라도 있는 것이 좋다. 또한 `__syncthreads()`를 기다리지 않는 블록이 하드웨어를 계속해서 바쁘게 유지하도록 여러 활성 블록들이 있어야 한다. 이는 리소스 가용성에 따라 결정되므로 두 번째 파라미터인 블록 당 스레드 수(블록 크기), 그리고 shared memory 사용량도 함께 고려해야 한다.

여러 개의 블록들이 동시에 멀티프로세서에 상주할 수 있기 때문에, 점유율은 블록 크기만으로 결정되지 않는다는 것을 기억하는 게 중요하다. 특히, 블록의 크기가 더 크다고 더 높은 점유율을 의마하지 않는다.

그리고, 점유율이 높다고 성능이 항상 향상되는 것은 아니다. 예를 들어, 점유율이 66%에서 100%로 향상될 때, 성능이 동일한 비율로 향상되는 것이 아니다. 더 낮은 점유율의 커널은 더 높은 점유율의 커널보다 사용 가능한 레지스터가 더 많다. 따라서, register spilling이 발생할 염려가 없다. 특히, 높은 수준의 instruction-level parallelism (ILP)를 통해 낮은 점유율로도 latency를 완전히 커버할 수 있다.

블록 크기를 선택할 때 관련되어 있는 요소들이 많고, 어쩔 수 없이 약간의 실험이 필요하다. 다만, 몇 가지 경험으로부터 얻을 수 있는 규칙들이 있다.

- 블록 당 스레드는 warp 크기의 배수이어야 한다.
- 멀티프로세서 당 여러 동시 블록들이 있는 경우에 블록 당 최소 64개의 스레드를 사용해야 한다.
- 블록 당 128~256개의 스레드는 다양한 블록 크기로 실험하기에 좋은 초기 범위이다.
- Latency가 성능에 영향을 미치는 경우, 멀티프로세서 당 하나의 큰 스레드 블록이 아닌 여러 개의 작은 스레드 블록을 사용해야 한다. 이는 `__syncthreads()`를 자주 호출하는 커널에 특히 유용하다.

> 스레드 블록이 멀티프로세서에서 사용할 수 있는 것보다 더 많은 레지스터를 할당하거나, 너무 많은 shared memory 또는 너무 많은 스레드가 요청되면 kernel launch가 실패할 수 있다.

<br>

# Effects of Shared Memory

Shared memory는 global memory의 중복 액세스를 제거하거나 coalesced access 패턴을 사용하는데 유용하다. 그러나 점유율을 제한할 수 있다. 대부분 shared memory를 사용할 때는 커널의 블록 크기와 연관이 되어 있는데, 스레드와 shared memory는 일대일 매핑이 아니어도 된다. 예를 들어, 커널에서 64x64 크기의 shared memory 배열을 사용하고 하나의 스레드에 하나의 요소를 매핑하려면 블록 당 스레드의 수가 64x64=4096이다. 블록 당 최대 스레드의 수는 1024개이므로 64x64 스레드 블록으로 커널을 실행할 수 없다. 이 경우에는 32x32개 또는 64x16개의 스레드로 구성된 커널을 실행하여 shared memory 배열의 4개 요소를 하나의 스레드가 처리하도록 할 수 있다.

Execution configuration의 세 번째 파라미터를 지정하여 동적으로 shared memory의 할당 크기를 조절할 수 있고, 이를 변경해가면서 점유율이 어떻게 영향을 주는지 실행해볼 수 있다 (커널 코드를 수정하지 않고 이 파라미터만 증가시켜서 점유율을 감소시킬 수 있고, 이에 따른 성능의 변화를 측정할 수 있음).

<br>

# Concurrent Kernel Execution

> Concurrent kernel execution에 대한 내용은 아래 포스팅에서 더 많은 내용을 확인할 수 있다.
> - [Asynchronous Concurrent Execution](/cuda/doc/01_programming_guide/03-02-08_asynchronous_concurrent_execution.md)
> - [Introducing CUDA Streams](/cuda/)

CUDA 스트림을 사용하면 kernel execution과 data trasfers를 오버랩시킬 수 있다. Concurrent kernel execution이 가능한 device에서, 스트림을 사용하면 여러 커널을 동시에 실행시켜 멀티프로세서를 보다 많이 활용할 수 있다. 이 기능이 가능한 지에 대한 여부는 `cudaDeviceProp` 구조체의 `concurrentKernels` 멤버를 통해 확인할 수 있다. 동시 실행에는 non-default 스트림(stream 0이 아닌 스트림)이 필요하다. Default 스트림을 사용하는 커널 호출은 모든 스트림에서의 이전 호출이 완료된 이후에 시작되며, 이후의 호출들은 default 스트림의 호출이 끝날 때까지 기다리게 된다.

아래 예제 코드는 기본적인 concurrent execution 기법을 보여준다. `kernel1`과 `kernel2`는 서로 다른 non-default 스트림에서 실행되며, 두 커널을 동시에 수행된다.
```c++
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
kernel1<<<grid, block, 0, stream1>>>(data_1);
kernel2<<<grid, block, 0, stream1>>>(data_2);
```

<br>

# Multiple Contexts

CUDA 작업은 context라고 하는 특정 GPU에 대한 프로세스 공간에서 발생한다. Context는 해당 GPU에 대한 kernel launches, 메모리 할당 및 page tables와 같은 지원 요소들을 캠슐화한다. CUDA Driver API에서는 context가 명시적으로 지정되지만, Runtime API에서는 context가 자동으로 생성되고 관리된다.

CUDA Driver API를 사용하면 CUDA 어플리케이션은 동일한 GPU에 대해 하나 이상의 context를 생성할 수 있다. 동일한 GPU에서 여러 CUDA 어플리케이션 프로세스에 액세스하는 경우는 대개 multiple contexts를 의미한다. 왜냐면 context는 Multi-Process Service를 사용하지 않는 한 특정 호스트 프로세스에 바인딩되기 때문이다.

특정 GPU에 대해 동시에 여러 context(그리고 관련된 리소스)를 할당할 수 있지만, 이러한 context 중 하나에서만 해당 GPU의 특정 시점에서 작업을 수행할 수 있다. 동일한 GPU를 공유하는 context는 시간으로 분할된다(time-sliced). 추가 context를 만드는 것은 context switching에 대한 time overhead와 context 마다 memory overhead를 유발한다. 또한, 여러 context에서 작업이 동시에 실행될 수 있는 경우에도 context switching이 필요하여 GPU utilization을 떨어뜨릴 수 있다.

따라서, 동일한 CUDA 프로그램 내에서는 하나의 GPU가 여러 context를 사용하는 것을 피하는 것이 좋다. 이를 지원하기 위해, CUDA Driver API에서는 각 GPU에 대한 특별한 primary context에 액세스하고 관리하기 위한 메소드들을 제공한다 (아래 예제 코드 참조). 이 context는 CUDA Runtime에서 host 스레드에서 사용하는 context가 없는 경우에 사용되는 것과 동일하다.

```c++
CUcontext ctx;
cuDevicePrimaryCtxRetain(&ctx, dev);

// When the program/library launches work
cuCtxPushCurrent(ctx);
kernel<<<...>>>(...);
cuCtxPopCurrent(&ctx);

// When the program/library is finished with the context
cuDevicePrimaryCtxRelease(dev);
```

> NVIDIA-SMI can be used to configure a GPU for [exclusive process mode](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-modes), which limits the number of contexts per GPU to one. This context can be current to as many threads as desired within the creating process, and `cuDevicePrimaryCtxRetain` will fail if a non-primary context that was created with the CUDA driver API already exists on the device.

<br>

# References

- [NVIDIA CUDA Documentation: Execution Configuration Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations)