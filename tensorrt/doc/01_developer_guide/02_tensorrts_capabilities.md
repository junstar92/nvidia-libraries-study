# Table of Contents

- [Table of Contents](#table-of-contents)
- [C++ and Python APIs](#c-and-python-apis)
- [The Programming Model](#the-programming-model)
- [Plugins](#plugins)
- [Types and Precision](#types-and-precision)
- [Quantization](#quantization)
- [Tensors and Data Formats](#tensors-and-data-formats)
- [Dynamic Shapes](#dynamic-shapes)
- [DLA](#dla)
- [Updating Weights](#updating-weights)
- [Streaming Weights](#tensorrt-100-straming-weights)
- [trtexec Tool](#trtexec-tool)
- [Polygraphy](#polygraphy)
- [References](#references)

<br>

# C++ and Python APIs

TensorRT API는 C++과 Python에서 모두 제공되며, 두 언어에서 거의 동일한 기능을 제공한다. 파이썬 API는 Numpy나 Scipy와 같은 파이썬 데이터 처리 라이브러리와 함께 사용될 수 있어서 매우 용이하다. C++ API는 더 효율적일 수 있으며, 자동차 어플리케이션에서 일부 요구 사항을 더 잘 충족할 수 있다.

> Python API는 모든 플랫폼에서 사용 가능한 것은 아니다. 이에 대한 정보는 [NVIDIA TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)를 참조.

<br>

# The Programming Model

TensorRT는 아래의 두 단계로 동작한다.

1. The Build Phase
2. The Runtime Phase

첫 번째 빌드 단계는 일반적으로 으프라인으로 수행되는데, TensorRT에 model definition을 제공하고 TensorRT는 이를 target GPU에 맞게 최적화한다. 두 번째 런타임 단계는 최적화된 모델을 사용하여 추론을 수행한다.

## The Build Phase

TensorRT에서 빌드 단계를 위한 highest-level interface는 `Builder`이다. Builder는 모델을 최적화하고 `Engine`(엔진)을 생성한다.

엔진을 빌드하기 위해서는 아래의 과정이 필요하다.

1. Create a network definition
2. Specify a configuration for the builder (precision, ...)
3. Call the builder to create the engine

`NetworkDefinition` 인터페이스는 모델을 정의한다. 가장 일반적인 방법은 학습 프레임워크로부터 ONNX 포맷으로 모델을 export하고, TensorRT의 ONNX parser를 통해 network definition을 생성하는 것이다. 물론, TensorRT의 `Layer`와 `Tensor` 인터페이스를 사용하여 step by step으로 network definitino을 구성할 수도 있다.

어떤 방법을 사용하든 네트워크의 input과 output인 텐서(tensor)를 정의해야 한다. Output으로 마킹되지 않은 텐서는 Builder에 의해서 최적화될 여지가 있는 임시 값으로 취급된다. Input과 output 텐서에는 반드시 이름이 지정되므로, 런타임 시 TensorRT는 input 및 output 버퍼를 모델에 바인딩하는 방법을 알 수 있다.

`BuilderConfig` 인터페이스는 TensorRT가 모델을 어떻게 최적화할 지를 지정하는데 사용된다. 옵션 중에는 연산의 precision 제어, 메모리와 실행 속도 사이의 tradeoff 제어, CUDA 커널 선택 제어 등을 제한할 수 있다. 모델 빌드 시간은 몇 분 이상이 소요될 수 있으므로, builder가 커널을 검색하는 방법과 검색 결과를 후속 실행에서 사용하기 위해 캐싱하도록 제어할 수도 있다.

`Network definition`과 `builder configuration`이 준비되었다면, builder를 호출하여 엔진을 생성할 수 있다. Builder는 dead computation을 제거, constant folding, 그리고 GPU에서 더 효율적으로 실행하기 위해 연산을 재정렬/결합한다. 그리고, 선택적으로 부동소수점 연산의 precision을 감소시켜, 16비트 부동소수점으로 실행하거나 quantization을 통해 8비트 정수로 실행할 수 있다. 또한, 각 레이어에서 다양한 data format(`NCHW`, `NHWC`, ...)에 대한 여러 구현의 실행 시간을 측정하고 모델을 실행하기 위한 최적의 스케줄을 계산하여 kernel execution과 format transformation의 총 비용을 최소화한다.

<br>

Builder는 **플랜(plan)** 이라고 부르는 serialized form의 엔진을 생성한다. 이는 바로 deserialize될 수 있거나, 파일로 저장할 수 있다.

> - TensorRT로부터 생성된 엔진은 기본적으로 생성된 TensorRT의 버전과 GPU에 따라 특정된다. 호환성을 위해 미리 엔진을 구성하는 방법은 [Version Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#version-compat)와 [Hardware Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#hardware-compat)를 참조 바람.
> - TensorRT의 network definition는 파라미터(such as the weights for a convonlution)를 deep-copy하지 않는다. 따라서, 빌드가 완료될 때까지 파라미터 배열의 메모리를 해제하지 말아야 한다. ONNX parser를 사용하는 경우, ONNX 파일로부터 네트워크를 import할 때, parser가 파라미터 배열을 소유하고 있다. 따라서, 빌드가 완료될 때까지 parser를 destroy하면 안된다.
> - Builder는 가장 빠른 알고리즘을 찾기 위해서 실행 시간을 측정한다. 만약 다른 GPU 작업과 병렬로 실행하면 실행 시간이 왜곡되므로 poor optimization이 될 수 있다.

## The Runtime Phase

Execution phase를 위한 highest-level 인터페이스는 `Runtime`이다.

런타임을 사용할 때, 일반적으로 다음의 단계를 실행해야 한다.

1. Deserialize a plan to create an engine
2. Create an execution context from the engine

그리고, 반복적으로 아래 단계들을 수행한다.

- Populate input buffers for inference
- Call `enqueueV3()` on the execution context to run inference

`Engine` 인터페이스는 최적화된 모델을 나타낸다. 사용자는 엔진을 쿼리하여 네트워크의 input/output 텐서의 정보(expected dimensions, data type, format, ...)를 얻을 수 있다.

`ExecutionContext` 인터페이스는 엔진으로부터 생성되며, 추론을 수행하는 기본 인터페이스이다. 여기에는 특정 호출과 관련된 모든 상태가 포함되어 있다. 따라서, 하나의 엔진과 연결된 여러 컨텍스트를 생성하여 가질 수 있으며, 이를 통해 병렬로 추론을 실행할 수 있다.

추론하기 위해 호출할 때, 적절한 위치에 input/output 버퍼를 설정해야 한다. 데이터 특성에 따라서 메모리는 CPU 또는 GPU에 있을 수 있다. 모델에 따라서 위치가 명확하지 않은 경우에는 엔진을 쿼리하여 버퍼에 제공할 메모리 공간을 결정할 수 있다.

버퍼가 모두 설정된 이후에는 추론을 큐에 넣을 수 있다 (`euqueueV3`). 요청된 커널은 CUDA stream의 큐에 들어가고, 제어권은 곧바로 반환된다. 일부 네트워크에서는 CPU와 GPU 간의 제어가 필요할 수 있으므로 제어권이 즉시 반환되지 않을 수도 있다. 만약, 비동기 실행이 완료될 때까지 대기하려면 `cudaStreamSynchronize`를 사용하여 스트림을 동기화시켜주면 된다.

> 방금 설명에서는 비동기로 추론을 실행하는 것만 언급했다 (`enqueueV3`). TensorRT 런타임에서는 default stream에서 실행하는 `executeV2` 인터페이스도 제공한다.

# Plugins

TensorRT에서는 기본적으로 지원하지 않는 연산들을 구현하기 위한 `Plugin` 인터페이스를 제공한다. 이렇게 구현된 플러그인들은 TensorRT의 `PluginRegistry`에 등록하여 사용할 수 있다. 예를 들어, ONNX parser로 ONNX 파일을 파싱하여 네트워크를 번역할 때, `PluginRegistry`에서 플러그인을 찾아서 사용할 수 있다.

TensorRT 자체에서 제공하는 몇 가지 플러그인들이 있으며, 이는 [link](https://github.com/NVIDIA/TensorRT/tree/main/plugin)에서 찾아볼 수 있다.

**[TensorRT 10.0]** `cuDNN`와 `cuBLAS`는 더 이상 TensorRT와 함께 제공되지 않기 때문에 별도로 해당 라이브러리를 설치해야 한다. `cudnnContext*` 또는 `cublasContext*`를 얻으려면 해당하는 `TacticSource` 플래그를 `nvinfer1::IBuilderConfig::setTacticSource()`를 사용하여 설정해주어야 한다.

당연히 사용자가 새로운 플러그인을 작성하여 사용할 수 있으며, 이에 대한 내용은 [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)에서 확인할 수 있다.

# Types and Precision

## Supported Types

TensorRT는 `FP32`, `FP16`, `INT8`, `INT32`, `UINT8`, `BOOL`의 데이터 타입을 지원한다. TensorRT 10.0부터 `BF16`, `FP8`, `INT4`, `INT64` 타입도 지원한다.

- `FP32`, `FP16`
  - Unquantized higher precision types
- **[TensorRT 10.0]** `BF16`
- `INT8`
  - Implicit quantization
  - Explicit quantization
- **[TensorRT 10.0]** `INT4` : low-precision integer type for weight compression
  - used for weight-only-quantization (requires dequantization before compute is performed).
  - conversion to and from INT4 type requires an explicit Q/DQ/ layer.
  - INT4 weight sare expected to be serialized by packing two elements per-byte (refer [Quantized Weights](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qat-weights)).
- **[TensorRT 10.0]** `FP8` : low-precision floating-point type
  - 8-bit floating point type with 1-bit for sign, 4-bits for exponent, 3bits for mantissa (4E3M FP8).
  - conversion to and from FP8 type requires an explicit Q/DQ layer.
- `UINT8` : only usable as a network I/O type (UINT8 quantization is not supported)
- `BOOL` : used with supported layers

## [TensorRT 10.0] Strong Typing vs Weak Typing

TensorRT에 네트워크를 제공할 때, strongly type인지 weakly type인지 지정할 수 있고, 기본적으로는 weakly type으로 지정된다.

Strongly typed networks에서 TensorRT의 optimizer는 network input 타입과 operator의 specification에 기반하여 중간 텐서 타입을 정적으로 추론한다. 자세한 내용은 [Strongly Typed Network](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#strongly-typed-networks) 참조.

Weakly typed networks에서 TensorRT의 optimizer는 성능을 높일 수 있는 경우에 다른 precision으로 대체할 수 있다. 이 모드에서 기본적으로 모든 부동소수점 연산은 `FP32`이다. 하지만 다른 수준의 precision을 선택하는 두 가지 방법이 있다.

- Model level에서 precision을 제어하려면, `BuilderFlag` 옵션을 사용하여 TensorRT가 더 빠른 구현을 찾도록 더 낮은 precision을 선택할 수 있도록 할 수 있다 (더 낮은 precision에서 일반적으로 더 빠르다).
  
  그러므로, 모델 전체에서 `FP16` 연산을 사용하도록 TensorRT에 쉽게 지시할 수 있다. 입력의 dynamic range가 약 1인 정규화된 모델의 경우, 정확도에 거의 영향을 미치지 않으면서 속도가 상당히 향상될 수 있다.
- 조금 더 세밀한 제어를 하고 싶다면, 해당 레이어에 대해서만 precision을 따로 지정할 수 있다. 예를 들어, 네트워크의 특정 부분은 수치적으로 민감할 수 있기 때문에, 이런 경우에는 해당 레이어만 더 높은 precision을 사용하도록 지정한다.

이에 대한 자세한 내용은 [Reduce Precision](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision)에서 설명하고 있다.

# Quantization

[Updated in TensorRT 10.0]

TensorRT는 선형적으로 압축되거나 반올림되는 low precision quantized types (INT8, FP8, INT4)의 quantized floating point를 지원한다. 이를 통해 산술처리량을 증가시키면서 storage 및 memory bandwidth를 줄일 수 있다.

TensorRT는 8비트 정수로 반올림되는 qunatized floating-point를 지원한다. 이를 통해 산술처리량을 증가시키면서 storage requirements 및 memory bandwidth를 줄일 수 있다. 부동소수점 텐서를 양자화할 때, TensorRT는 해당 텐서의 dynamic range를 알고 있어야 한다. 해당 범위 밖의 값은 clamping 된다.

Dynamic range 정보는 대표되는 input data를 기반으로 builder가 계산할 수 있다. 이 과정을 TensorRT에서는 `calibration`이라고 부른다. 또는, 프레임워크에서 quantization-aware training(QAT)를 수행하고 필요한 dynamic range 정보와 함께 모델을 TensorRT로 가져올 수도 있다.

[Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)에서 조금 더 자세한 내용을 살펴볼 수 있다.

# Tensors and Data Formats

네트워크를 정의할 때, TensorRT는 텐서가 multidimensional C-style 배열이라고 가정한다. 각 레이어마다 입력에 대한 특정 해석이 있는데, 예를 들어, 2D convolution layer는 입력의 마지막 3개의 차원이 `CHW` 포맷이라고 가정하며, `WHC` 포맷을 사용할 수 있는 옵션은 존재하지 않는다.

> 텐서의 요소 수는 최대 $2^{31}-1$로 제한된다.

네트워크를 최적화하는 동안 TensorRT는 가능한 가장 빠른 CUDA 커널을 선택할 때, 내부적인 변환(`HWC`뿐만 아니라 더 많은 포맷이 있다)을 수행한다. 일반적으로 이러한 포맷은 성능을 최적화하기 위해 내부적으로 선택되며 어플리케이션은 이를 제어할 수 없다. 그러나 불필요한 변환을 최소화할 수 있도록 기본적인 데이터 포맷(underlying data format)은 I/O boundaries (network input/output, and passing data to/from plugins)에서 포맷을 제어할 수 있다.

이에 대한 자세한 내용은 [I/O Formats](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors)에서 확인할 수 있다.

# Dynamic Shapes

기본적으로 TensorRT는 모델이 정의된 input shapes(batch size, image size, ...)를 기반으로 모델을 최적화한다. 즉, 지정된 크기에 대해 최적화를 수행한다. 그러나 런타임 시에, 즉, 어플리케이션에서 추론할 때, input dimensions를 builder를 통해 조정할 수 있다. 이를 활성화하려면 builder configuration에서 `OptimizationProfile` 인스턴스를 최소 하나 이상 지정해야 하며, `OptimizationProfile`에는 해당 범위 내의 최적화 지점과 함께 각 입력에 대한 최소/최대 shape를 포함한다.

TensorRT는 [minimum, maximum] 범위 내의 모든 shape에 대해 동작하며, 해당 최적화 지점에서 가장 빠른 CUDA 커널(일반적으로 profile마다 다르다)을 선택하여 각 profile에 대해 최적화된 엔진은 생성한다. 그런 다음, 런타임에 이 profile 중에서 하나를 선택하여 추론을 수행할 수 있다.

이에 대한 자세한 내용은 [Working with Dynamic Shapes](hhttps://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)에서 설명하고 있다.

# DLA

TensorRT는 NVIDIA의 Deep Learning Accelerator(DLA)를 지원한다. DLA는 NVIDIA SoC에 존재하는 추론 전용 프로세서이다. DLA는 TensorRT의 일부 제한된 레이어들을 지원하며, TensorRT는 네트워크 일부를 DLA에서 실행하고 지원되지 않는 나머지 레이어에서는 GPU에서 실행하도록 할 수 있다. 두 장치에서 모두 실행할 수 있는 레이어는 builder configuration에서 각 레이어마다 하나를 지정할 수 있다.

이에 대한 자세한 내용은 [Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla_topic)에서 다루고 있다.

# Updating Weights

엔진을 빌드할 때, 나중에 weight를 업데이트하도록 지정할 수 있다. 이는 강화학습에서와 같이 네트워크 구조는 변경하지 않으면서 모델의 weight만 자주 업데이트하거나 동일한 구조를 유지하면서 모델을 재학습하는 경우에 유용할 수 있다. Weight 업데이트는 `Refitter` 인터페이스를 사용하여 수행한다.

이에 대한 자세한 내용은 [Refitting an Engines](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c)에서 다루고 있다.

# [TensorRT 10.0] Straming Weights

TensorRT는 engine load time에 가중치를 device memory에 로드하는 것이 아닌, 네트워크를 실행할 때 가중치를 host memory에서 device memory로 스트림할 수 있는 기능을 제공한다. 이를 통해 제한된 GPU 크기보다 더 큰 메모리의 가중치를 가진 모델을 실행할 수 있지만, 잠재적으로 latency가 상당히 증가한다. 이 기능은 build time (`BuilderFlag::kWEIGHT_STREAMING`)과 runtime (`ICudaEngine::setWeightStreamingBudget`) 모두에서 가능하다.

# trtexec Tool

samples 디렉토리에는 TensorRT를 사용하지 않으면서 자체 어플리케이션을 개발할 필요없이 사용할 수 있는 command-line wrapper tool인 `trtexec`가 포함되어 있다. `trtexec`는 다음의 세 가지 목적을 가지고 있다.

- 임의의 입력 데이터 또는 사용자 제공 데이터에서 네트워크를 벤치마킹한다.
- 주어진 모델로부터 serialized engines를 생성한다.
- Builder로부터 serialized timing cache를 생성한다.

`trtexec`에 대한 내용은 [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)에서 자세히 다루고 있다.

# Polygraphy

Polygraphy는 TensorRT 및 다른 프레임워크에서 딥러닝 모델을 실행하고 디버깅하는데 도움이 되는 툴킷이다. 여기에는 [Python API](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)과 [command-line interface(CLI)](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools)를 포함한다. Polygraphy를 사용하면 아래의 작업들을 수행할 수 있다.

- TensorRT 및 ONNX-Runtime과 같은 여러 백엔드에서 추론을 수행하고 결과를 비교한다 (ex, [API](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/api/01_comparing_frameworks), [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/run/01_comparing_frameworks)).
- 모델을 다양한 포맷으로 변환한다. 예를 들어, Post-Training Quantization(PTQ)가 적용된 TensorRT 엔진이 있다 (ex, [API](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/api/04_int8_calibration_in_tensorrt), [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/convert/01_int8_calibration_in_tensorrt)).
- 다양한 타입의 모델에 대한 정보를 쿼리한다 (ex, [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect)).
- 커맨드라인에서 ONNX 모델을 수정한다.
  - Extract subgraphs (ex, [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/surgeon/01_isolating_subgraphs))
  - Simplify and sanitize (ex, [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants))
- Isolate faulty tactics in TensorRT (ex, [CLI](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics))

Polygraphy에 대한 내용은 [Polygraph repository](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)에서 살펴볼 수 있다.

# References

- [NVIDIA TensorRT Documentation: TensorRT's Capabilities](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fit)
- [TensorRT Plugins (Github)](https://github.com/NVIDIA/TensorRT/tree/main/plugin)
- [Polygraph (Github)](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)