# Table of Contents

- [Table of Contents](#table-of-contents)
- [Object Lifetime](#object-lifetime)
- [Error Handling and Logging](#error-handling-and-logging)
- [Memory](#memory)
  - [The Build Phase](#the-build-phase)
  - [The Runtime Phase](#the-runtime-phase)
  - [CUDA Lazy Loading](#cuda-lazy-loading)
  - [L2 Persistent Cache Management](#l2-persistent-cache-management)
- [Threading](#threading)
- [Determinism](#determinism)
  - [IFillLayer Determinism](#ifilllayer-determinism)
- [Runtime Options](#runtime-options)
- [Compatibility](#compatibility)
- [References](#references)

<br>

# Object Lifetime

TensorRT의 API는 클래스 기반이며, 몇몇 클래스는 다른 클래스의 팩토리 클래스처럼 동작한다. 유저가 소유하는 객체의 경우, 팩토리 객체의 수명은 팩토리 객체가 생성하는 객체의 수명과 같아야 한다. 예를 들어, `NetworkDefinition` 및 `BuilderConfig` 클래스는 `Builder` 클래스로부터 생성되며, 생성된 두 객체는 builder 팩토리 객체가 파괴되기 전에 파괴되어야 한다.

이러한 규칙의 예외는 builder로부터 엔진을 생성하는 경우이다. 엔진을 생성한 이후에는 builder, network, parser, build config를 제거해도 엔진을 계속 사용할 수 있다.

<br>

# Error Handling and Logging

TensorRT의 top-level 인터페이스(builder, runtime or refitter)를 생성할 때, `Logger` 인터페이스의 구현을 반드시 제공해야 한다. 로거(logger)는 진단 및 정보들에 대한 메세지를 위해 사용되며, 출력 수준(verbosity level)은 제어 가능하다. TensorRT의 lifetime의 어느 시점에서든 정보를 전달하는데 사용될 수 있으므로, 로거의 수명은 어플리케이션에서 해당 인터페이스를 사용하는 모든 구간을 포함하고 있어야 한다. TensorRT가 내부적으로 worker threads를 사용할 수 있으므로, 로거의 구현은 thread-safe해야 한다.

객체에서의 API 호출은 해당 객체의 상위 인터페이스와 연관된 로거를 사용한다. 예를 들어, `ExecutionContext::enqeueV3()` 호출에서 execution context는 엔진으로부터 생성되었고, 엔진은 런타임으로부터 생성되었으므로 TensorRT는 해당 런타임과 관련된 로거를 사용한다.

에러를 처리하는 기반 방법은 `ErrorRecorder` 인터페이스이다. 이 인터페이스를 구현한 뒤, API 객체에 연결하여 해당 객체와 관련된 오류를 받을 수 있다. 객체의 에러 레코더(error recorder)는 해당 객체가 생성하는 다른 객체에도 전달된다. 예를 들어, 엔진에 에러 레코더를 연결하고, 해당 엔진에서 execution context를 생성하면 동일한 레코더를 execution context에서 사용한다. 만약 에러 레코더를 execution context에 연결하면, 에러 레코더는 해당 execution context의 에러만 받는다. 에러가 발생했지만 연결된 에러 레코더가 없으면 연결된 로거를 통해 전송된다.

> CUDA 에러는 일반적으로 비동기다. 따라서, 다른 여러 추론을 동시에 수행하거나 다른 CUDA 스트림이 하나의 CUDA Context에서 동작할 때, 비동기적인 GPU 에러는 에러가 발생한 위치가 아닌 다른 위치에서 관측될 수도 있다.

<br>

# Memory

TensorRT는 상당한 크기의 device memory를 사용한다. Device memory는 제한적이므로 TensorRT가 메모리를 사용하는 방식을 이해하는 것이 중요하다.

## The Build Phase

빌드 단계에서 TensorRT는 레이어 구현의 시간을 측정하기 위한 device memory를 할당한다. 일부 구현에서는 많은 양의 임시 메모리를 사용할 수 있다. 이러한 임시 메모리의 크기는 builder config의 memory pool limit를 통해 제어할 수 있다. 기본 workspace의 크기는 device의 전체 global memory 크기이지만, 필요한 경우에는 이를 제한할 수 있다. 만약 builder가 workspace가 부족하여 실행할 수 없는 커널을 찾으면 이를 로깅 메세지를 통해 알려준다.

비교적 workspace가 작더라도 timing은 inputs, outputs, weights에 대한 버퍼 생성을 필요로 한다. TensorRT는 OS가 이러한 할당에 대해 out-of-memory를 반환하는 것에 대해 강력하다. 일부 플랫폼에서 OS는 성공적으로 메모리를 할당할 수 있는데, 이 경우 out-of-memory killer 프로세스가 시스템에 메모리가 부족하다고 리포트하고 TensorRT를 종료한다. 만약 이러한 경우가 발생하면 최대한 많은 시스템 메모리를 확보하고 난 뒤, 재시도하는 것이 좋다.

빌드 단계에서는 일반적으로 적어도 두 개의 weight 복사본이 host memory에 존재한다. 하나는 original network에서 가져온 것이고, 다른 하나는 엔진을 빌드하는 동안 포함되는 것이다. 또한, TensorRT가 weights를 결합할 때, 추가적인 임시 weight 텐서가 생성된다 (ex, convolution with batch normalization).

## The Runtime Phase

런타임에서 TensorRT는 host memory는 비교적 적게 사용하고, device memory는 상당히 많이 사용한다.

엔진은 역직렬화(deserialization)할 때 모델의 weights를 저장하기 위해 device memory를 할당한다. 직렬화된 엔진(serialized engine)의 대부분은 weights 정보이기 때문에 직렬화된 엔진의 크기는 weight를 저장하는데 필요한 device memory 크기에 근사된다.

`ExecutionContext`는 두 종류의 device memory를 사용한다:

- 일부 레이어 구현에 필요한 영구적인 메모리 - 예를 들어, 일부 convolution layer는 edge masks를 사용한다. 이는 weights와 달리 여러 context에서 공유할 수 없는데, 이 크기는 context마다 input shape가 다를 수 있기 때문이다. 이 메모리는 execution context를 생성할 때 할당되며, execution context의 수명동안 지속된다.
- 중간 결과를 유지하는 데 사용되는 임시 메모리 - 이 메모리는 intermediate activation tensors에 사용된다. 또한, 레이어 구현에 필요한 임시 저장소로도 사용되며, 이 메모리의 bound는 `IBuilderConfig::setMemoryPoolLimit()` 함수로 제어된다.

`ICudaEngine::createExecutionContextWithoutDeviceMemory()` 함수를 사용하면 임시 메모리없이 execution context를 생성하고, 네트워크가 실행되는 동안 메모리를 직접 제공할 수도 있다. 이렇게 하면 동시에 실행되지 않는 여러 context 간 메모리를 공유할 수 있고, 또는 추론이 실행되는 않는 동안 다른 용도로도 사용할 수 있다. 필요한 임시 메모리 크기는 `ICudaEngine::getDeviceMemorySize()` 함수로 반환된다.

Execution context에서 사용되는 영구 메모리와 임시 메모리의 크기에 대한 정보는 네트워크를 빌드할 때, builder를 통해 출력된다 (로거의 출력 레벨이 `kINFO`인 경우). 로그 메세지는 다음과 같다. 단위는 바이트(bytes)이다.
```
[08/12/2021-17:39:11] [I] [TRT] Total Host Persistent Memory: 106528
[08/12/2021-17:39:11] [I] [TRT] Total Device Persistent Memory: 29785600
[08/12/2021-17:39:11] [I] [TRT] Total Scratch Memory: 9970688
```

기본적으로 TensorRT는 CUDA로부터 직접 메모리를 할당한다. 그러나, TensorRT의 `IGpuAllocator`([c++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_gpu_allocator.html)) 인터페이스 구현을 builder나 runtime에 제공하고 device memory를 직접 관리할 수 있다. 이는 TensorRT가 CUDA에서 직접 메모리를 할당하는 대신 어플리케이션이 모든 GPU 메모리를 제어하고 TensorRT에 suballocation하는 경우에 유용하다.

TensorRT의 종속성(cuDNN, cuBLAS)은 많은 양의 device memory를 차지할 수 있다. TensorRT를 사용하면 builder configuration에서 `TacticSources` 속성을 사용하여 이러한 라이브러리들이 추론에서 사용되는지 여부를 제어할 수 있다. 일부 플러그인 구현에셔는 이러한 라이브러리가 꼭 필요하므로, 이 라이브러리들이 제외된다면 네트워크가 성공적으로 컴파일되지 않을 수 있다.

또한, `PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805`는 TensorRT core 라이브러리에서 cuDNN, cuBLAS 및 cuBLASLt의 사용을 제어하는데 사용된다. 이 플래그가 설정되면 TensorRT core 라이브러리는 `IBbuilderConfig::setTacticSources()`에 지정된 경우에도 이러한 tactic을 사용하지 않는다. 이 플래그는 적절한 tactic sources가 설정된 경우 `IPluginV2Ext::attachToContext()`를 사용하여 플러그인에 전달된 `cudnnContext` 및 `cublasContext` 핸들에 영향을 미치지 않는다. **이 플래그는 기본적으로 설정(set)된다.**

CUDA infrastructure와 TensorRT의 device code도 device memory를 소비한다. 이 메모리 양은 플랫폼, GPU 및 TensorRT 버전에 따라 다르다. 이 메모리 크기는 `cudaGetMemInfo`를 사용하여 사용 중인 GPU 장치의 메모리 총량을 확인할 수 있다.

TensorRT는 builder와 runtime에서 중요한 작업 전후에 사용 중인 메모리 양을 측정한다. 이러한 메모리 사용 통계는 로거(Logger)에 의해 출력된다. 예를 들면, 아래와 같이 출력된다.
```
[MemUsageChange] Init CUDA: CPU +535, GPU +0, now: CPU 547, GPU 1293 (MiB)
```
위 출력은 CUDA initialization에 의해서 변경되는 메모리 사용을 나타낸다. `CPU +535, GPU+0`은 CUDA initialization이 실행된 이후 증가된 메모리 양을 나타낸다. `now:` 이후의 내용은 CUDA initialization 이후의 CPU/GPU 메모리 사용량 스냅샷이다.


## CUDA Lazy Loading

CUDA lazy loading은 TensorRT의 최대 GPU 및 host memory 사용량을 크게 줄이고 TensorRT 초기화를 가속화할 수 있는 CUDA의 기능이다. 성능에 대한 영향은 무시할 수 있는 수준이다(<1%). 메모리 사용량과 초기화 시간의 절약은 모델, 소프트웨어 스택, GPU 플랫폼 등에 따라 다르다. 이 기능은 `CUDA_MODULE_LOADING=LAZY`를 설정하여 활성화할 수 있다. 이에 대한 내용은 CUDA 문서([link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading))에서 확인할 수 있다.

## L2 Persistent Cache Management

NVIDIA Ampere와 이후의 아키텍처는 L2 cache persistence를 지원한다. 이 기능은 cache line이 제거될 때, L2 cache line의 우선순위를 지정할 수 있다. TensorRT는 이 기능을 사용하여 activations을 캐시에 유지하여 DRAM 트래픽과 전력 소비를 줄일 수 있다.

캐시 할당(cache allocation)는 execution context별로 수행되며 컨텍스트의 `setPersistentCacheLimit` 메소드로 활성화된다. 모든 컨텍스트(및 이 기능을 사용하는 다른 컴포넌트)의 persistent cache 총 합은 `cudaDeviceProp::persistingL2CacheMaxSize`를 초과하면 안된다.

> 이에 대한 내용은 CUDA 문서([link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#l2-cache))에서 다루고 있다.

<br>

# Threading

기본적으로 TensorRT 객체는 thread-safe 하지 않다. 다른 스레드로부터의 객체 액세스는 클라이언트에 의해 반드시 직렬화되어야 한다.

Runtime concurrency model은 서로 다른 스레드가 서로 다른 execution contexts에서 동작한다는 것이다. 컨텍스트에는 실행 중에 네트워크의 상태(activation values, ...)가 포함되므로 여러 스레드에서 컨텍스트를 동시에 사용하면 undefined behavior이 발생한다.

동시성 모델을 지원하기 위해, 아래의 연산들은 thread-safe 하다.

- Nonmodifying operations on a runtime or engine.
- Deserializing an engine from a TensorRT runtime.
- Creating an execution context from an engine.
- Registering and deregistering plugins.

서로 다른 스레드에서 여러 builder를 사용하는 경우에는 thread-safe 문제가 없다. 그러나 builder는 timing을 사용하여 제공된 파라미터에서 가장 빠른 커널을 결정하므로, 동일한 GPU로 여러 builder를 사용하면 최적의 엔진을 구성하는 timing과 TensorRT의 기능이 방해받는다. 서로 다른 GPU로 빌드하려고 여러 스레드를 사용하는 것은 문제되지 않는다.

<br>

# Determinism

TensorRT의 builder는 timing을 사용하여 주어진 연산을 구현하는 가장 빠른 커널을 찾는다. 커널의 시간을 측정하는 것은 노이즈에 따라 달라질 수 있다. 예를 들어, GPU에서 다른 작업이 수행중일 때나 GPU의 클럭 속도가 변동할 때가 있다. 이러한 노이즈로 인하여 빌드할 때마다 같은 구현이 선택되지 않을 수 있다. 일반적으로 다른 구현에서는 다른 순서의 부동소수점 연산은 사용하므로, 결과가 조금 다를 수 있다. 최종 결과에 미치는 영향은 미미하다. 그러나, 여러 정밀도(precision)을 사용하여 최적화하도록 구성되어 있다면, FP16과 FP32 간의 차이는 상당할 수 있다. 특히 네트워크가 정규화되어 있지 않거나, 수치적으로 더 민감한 경우에 특히 더 그렇다.

이외에도 다른 configuration options에 의해서 결과가 달라질 수 있다. 예를 들면, input size(ex, batch size) 또는 input profile에 대한 다른 optimization point가 있다 ([Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes) 참조).

`AlgorithmSelector` 인터페이스는 builder가 주어진 레이어에 대해 특정 구현을 선택하도록 만들 수 있다. 이를 사용하면 builder가 실행될 때마다 동일한 커널이 선택되도록 하는 것이 가능하다. 이에 대한 내용은 [Algorithm Selection and Reproductible Builds](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select)를 참조하길 바란다.

엔진이 빌드된 이후에는 `IFillLayer`를 제외하고, 모든 것이 결정적이다. 즉, 동일한 input에 대해서 동일한 output을 출력한다.

## IFillLayer Determinism

`RANDOM_UNIFORM` 또는 `RANDOM_NORMAL` 연산을 사용하는 `IFillLayer`를 네트워크에 추가하면, 더 이상 determinism이 보장되지 않는다. 각 호출마다 이러한 연산은 RNG 상태를 기반의 텐서를 생성하고, RNG 상태를 업데이트한다. 이 상태는 execution context별로 저장된다.

<br>

# Runtime Options

TensorRT는 다양한 use cases를 충족하기 위해 많은 런타임 라이브러리를 제공한다. TensorRT 엔진을 실행하는 C++ 어플리케이션은 아래의 라이브러리 중 하나에 대해 링크해야 한다.

- The _default_ runtime is the main library (`libnvinfer.so/.dll`)
- The _lean_ runtime library(`libnvinfer_lean.so/.dll`) is much smaller than the default library, and contains only the code necessary to run a version-compatible engine. It has some restrictions; primarily, it cannot refit or serialize engines.
- The _dispatch_ runtime(`libnvinfer_dispatch.so/.dll`) is a small shim library that can load a lean runtime, and redirect calls to it. The dispatch runtime is capable of loading older versions of the lean runtime, and together with appropriate configuration of the builder, can be used to provide compatibility between a newer version of TensorRT and an older plan file. Using this dispatch runtime is almost the same as manually loading the lean runtime, but it checks that APIs are implemented by the lean runtime loaded, and performs some parameter mapping to support API changes where possible.

Lean runtime은 default runtime 보다 더 적은 연산 구현을 포함한다. TensorRT는 빌드 타임에 연산 구현을 선택하므로 버전 호환성을 활성화하여 lean runtime용으로 빌드되도록 지정해야 한다. 이렇게 빌드하면 default runtime에서 빌드한 엔진보다 약간 더 느릴 수 있다.

Lean runtime은 dispatch runtime의 모든 기능들을 포함한다. 그리고, default runtime은 lean runtime의 모든 기능들을 포함한다.

> C++용 runtime 라이브러리에 대응하는 파이썬 패키지는 다음과 같다.
> - `tensorrt` - It is the Python interface for the _default_ runtime.
> - `tensorrt_lean` - It is the Python interface for the _lean_ runtime.
> - `tensorrt_dispatch` - It is the Python interface for the _dispatch_ runtime.

<br>

# Compatibility

기본적으로 serialized engines는 오직 빌드가 직렬화된 시스템과 같은 OS, 같은 CPU architectures, 같은 GPU model, 같은 TensorRT 버전에서만 올바르게 동작한다. [Version Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#version-compat)와 [Hardware Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#hardware-compat)에서는 TensorRT 버전과 GPU 모델에 대한 제약을 완화하는 방법에 대해서 다루고 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work)