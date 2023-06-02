# Table of Contents

- [Table of Contents](#table-of-contents)
- [Version Compatibility](#version-compatibility)
  - [Manually Loading the Runtime](#manually-loading-the-runtime)
  - [Loading from Storage](#loading-from-storage)
  - [Using Version Compatibility with the ONNX Parser](#using-version-compatibility-with-the-onnx-parser)
- [Hardware Compatibility](#hardware-compatibility)
- [Compatibility Checks](#compatibility-checks)
- [References](#references)

<br>

# Version Compatibility

기본적으로 TensorRT 엔진은 오직 엔진이 빌드된 TensorRT 버전에서만 호환된다. 하지만 적절한 build-time configuration을 통해 TensorRT의 major 버전 내에서 minor 버전들 간의 호환이 되도록 빌드할 수 있다. 또한, TensorRT 8에서 빌드된 엔진은 TensorRT 9의 런타임에서 호환되는데, 반대로는 불가능하다.

Version compatibility는 8.6 버전부터 지원된다. 즉, plan은 적어도 8.6 버전 이상에서 빌드되어야만 하며, runtime 또한 8.6 버전 이상이어야 한다.

Version compatibility를 사용할 때, 런타임 시 지원되는 엔진의 API는 엔진이 빌드된 버전에서 지원하는 API와 엔진을 실행하는 버전의 API간의 교집합니다. TensorRT는 오직 major 버전 간에서만 API를 삭제하므로 major 버전 내에서는 API에 대해 걱정할 필요가 없다. 하지만 TensorRT 8 엔진을 TensorRT 9에서 사용할 때는 반드시 deprecated API에서 마이그레이션해야 한다.

Version-compatible 엔진을 생성하는데 권장되는 방법은 다음과 같다.
```c++
config->setFlag(BuilderFlag::kVERSION_COMPATIBLE);
IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
```

이 플래그는 implicit batch mode에서는 지원되지 않으며, network는 반드시 `NetworkDefinitionCreationFlag::kEXPLICIT_BATCH`로 생성되어야 한다.

Version Compatiblity를 적용하면 플랜(plan; serialized engine)에 lean runtime의 복사본이 추가된다. 이 플랜을 나중에 역직렬화할 때, TensorRT는 플랜에 lean runtime의 복사본이 있다는 것을 인식한다. 이는 lean runtime을 로드하고 역직렬화하는데 사용하여 플랜의 나머지 부분을 실행하는데 사용한다. 이로 인해 코드(code)가 owning process의 컨텍스트에서 플랜으로부터 로드되고 실행되므로 신뢰하는 플랜만 이 방법으로 역직렬화해야 한다. TensorRT가 플랜을 신뢰하도록 하려면 다음과 같이 호출하면 된다.
```c++
runtime->setEngineHostCodeAllowed(true);
```

> [setEngineHostCodeAllowed()](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#a5a19c2524f74179cd9b781c6240eb3ce)에서는 이 함수를 다음과 같이 설명하고 있다.
> 
> : Set whether the runtime is allowed to deserialize engines with host executable code.

> 플랜에 lean runtime이 포함된다는 말의 의미가 빌드된 환경의 TensorRT lean runtime 라이브러리가 포함된다는 것인지 확실하진 않다. 다만, 아래의 내용에 따르면 lean runtime 자체가 포함된다는 것으로 이해된다.

Trusted plan을 위한 이 플래그는 플랜 내에 플러그인(plugin)을 패키징할 때도 요구된다 ([Plugin Shared Libraries](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugin-serialization) 참조).

## Manually Loading the Runtime

방금 위에서 언급한 접근 방식은 모든 플랜의 런타임의 복사본을 패키징하므로, 어플리케이션에서 많은 모델들을 사용한다면 금지될 수 있다. 또 다른 방법은 런타임 로드를 직접 관리하는 것이다. 이 방법을 통해 위에서 설명한 version compatibile plans를 빌드하지만 lean runtime을 제외하도록 추가적인 플래그를 설정한다.
```c++
config->setFlag(BuilderFlag::kVERSION_COMPATIBLE);
config->setFlag(BuilderFlag::kEXCLUDE_LEAN_RUNTIME);
IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
```

이렇게 생성되노 플랜을 실행하려면, 반드시 플랜이 빌드된 버전의 lean runtime에 액세스해야 한다. TensorRT 8.6 버전에서 빌드된 플랜을 사용하고, 어플리케이션은 TensorRT 9에 대해 링킹한다고 가정한다면, 플랜을 다음과 같이 로드할 수 있다.
```c++
IRuntime* v9Runtime = createInferRuntime(logger);
IRuntime* v8ShimRuntime = v9Runtime->loadRuntime(v8RuntimePath);
engine = v8ShimRuntime->deserializeCudaEngine(v8plan);
```

런타임은 TensorRT 9 API 호출을 TensorRT 8.6 runtime 용으로 변환하고 해당 호출이 지원되는지 체크 및 필요한 파라미터 재배치를 수행한다.

## Loading from Storage

대부분의 OS에서 TensorRT는 메모리로부터 shared runtime 라이브러리를 직접 로드한다. 그러나, linux kernel 3.17 이전에서는 임시 디렉토리가 필요하다. `IRuntime::setTempfileControlFlags`와 `IRuntime::setTemporaryDirectory` APIs를 사용하여 이 메커니즘의 TensorRT 사용을 제어할 수 있다.

## Using Version Compatibility with the ONNX Parser

TensorRT의 ONNX parser로부터 생성된 network definition을 통해 version-compatible engines를 빌드할 때, 반드시 parser가 플러그인 대신 native `InstanceNormalization` 구현을 사용하도록 해야 한다.

이를 위해서 `IParser::setFlag()` API를 사용한다.
```c++
auto* parser = nvonnxparser::createParser(*network, *config);
parser->setFlag(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);
```

추가로, 네트워크에서 사용되는 ONNX 연산자들의 완벽한 구현을 위해 플러그인이 필요할 수 있다. 특히, 네트워크가 version-compatible engine을 빌드하는데 사용된다면, 몇몇 플러그인은 엔진과 함께 포함될 필요가 있다 (엔진과 함께 직렬화되거나 외부에서 제공 및 명시적으로 로드됨).

특정 파싱한 네트워크를 구현하는데 필요한 플러그인 라이브러리 리스트를 쿼리하려면, `IParser::getUsedVCPluginLibraries` API를 사용하면 된다.
```c++
auto* parser = nvonnxparser::createParser(*network, *config);
parser->setFlag(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);
parser->parseFromFile(filename, static_cast<int>(ILogger::Severity::kINFO));
int64_t nbPlulginLibs;
char const* const* pluginLibs = parser->getUsedVCPluginLibraries(nbPluginLibs);
```

[Plugin Shared Libraries](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugin-serialization)를 참조하면, 이렇게 쿼리한 라이브러리 리스트를 사용하여 플러그인을 직렬화하거나 외부에서 패키징하는 방법을 알 수 있다.

<br>

# Hardware Compatibility

기본적으로 TensortRT 엔진은 엔진이 빌드된 device 타입에서만 호환된다. 하지만 Build-configuration을 통해 다른 타입의 device에서도 호환되도록 빌드할 수 있다. 현재 hardware compatibility는 ampere 아키텍처와 그 이후의 아키텍처에서만 지원되며, Drive OS나 Jetson에서는 지원되지 않는다.

예를 들어, ampere 아키텍처와 그 이후의 아키텍처에서 호환되도록 빌드하려면, `IBuilderConfig`를 사용하여 다음과 같이 구성한다.
```c++
config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
```

Hardware compatibility를 사용하여 빌드할 때, TensorRT는 하드웨어 간 호환되지 않는 tactics은 제외한다. 이러한 tactics은 특정 아키텍처의 명령어를 사용하거나 다른 장치에서 사용 가능한 shared memory보다 더 많은 메모리를 요구하기 때문이다. 따라서, hardware compatible 엔진은 그렇지 않은 엔진보다 더 낮은 처리량과 더 높은 latency를 가질 수 있다. 성능 차이의 정도는 네트워크 아키텍처와 입력 크기에 따라 다르다.

<br>

# Compatibility Checks

TensorRT는 플랜(plan; serialized engine)에 플랜을 생성할 때 사용한 라이브러리의 major, minor, patch, build 버전을 기록한다. 만약 역직렬화(deserialization)할 때 사용되는 런타임의 버전과 플랜에 기록된 버전이 일치하지 않으면, 역직렬화는 실패한다. Version compatibility를 사용하는 경우, 플랜 데이터를 역직렬화하는 lean runtime에서 검사를 수행한다. 기본적으로 lean runtime은 플랜에 포함되며, 버전 일치 검사가 성공하도록 보장한다.

TensorRT는 플랜에 compute capability(major and minor) 버전 또한 기록하며, 플랜이 로드되는 GPU에 대해서 체크한다. 만약 일치하지 않으면 역직렬화는 실패한다. 이렇게 함으로써 빌드 과정에서 선택된 커널이 런타임에서 실행될 수 있도록 보장한다. Hardware compatibility를 사용하면, 이 검사가 완화된다.

TensorRT는 추가적으로 아래의 내용들에 대한 호환성 체크를 수행한다. 만약 일치하지 않는다면 경고를 출력한다 (hardware compatibility를 사용하는 경우는 제외).

- Global memory bus width
- L2 cache size
- Maximum shared memory per block and per multiprocessor
- Texture alignment requirement
- Number of multiprocessors
- Whether the GPU device is integrated or discrete

만약 엔진이 직렬화된 시스템과 런타임 시스템에서의 GPU 클락 속도가 다르다면, 직렬화할 때 선택된 tactics가 런타임 시스템에서 최적이 아닐 수 있다. 따라서, 성능 하락이 발생할 수도 있다.

또한, 역직렬화하는 동안 사용 가능한 device memory가 직렬화할 때 필요했던 크기보다 작다면, 역직렬화는 memory allocation failures로 실패할 수 있다.

만약 동일한 아키텍처의 여러 GPU 장치를 사용하여 하나의 TnesorRT 엔진을 최적화한다면, 추천하는 방법은 가장 작은 장치에서 builder를 실행하는 것이다. 또는, 제한된 리소스의 더 큰 장치로 엔진을 빌드할 수 있다 ([Limiting Compute Resources](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#limit-compute-resources) 참조). 이는 더 큰 GPU 장치에서 작은 모델을 빌드할 때, TensorRT는 덜 효율적이지만 가능한 리소스 내에서 확장성이 좋은 커널을 선택할 수도 있기 때문이다. 게다가, TensorRT가 cuDNN과 cuBLAS로부터 커널을 선택하고 구성하는데 사용하는 APIs는 장치간 호환성을 지원하지 않는다. 따라서, builder configuration에서 이러한 tactic 소스 사용을 비활성화한다.

Safety runtime은 경우에 따라 TensorRT의 major, minor, patch, build 버전이 정확하게 일치하지 않는 환경에서 생성된 엔진을 역직렬화할 수 있다. 이에 대한 더 많은 정보는 NVIDIA DRIVE OS 6.0 Developer Guide를 참조하면 된다.

<br>

# References

- [NVIDIA TensorRT Documentation: Version Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#version-compat)
- [NVIDIA TensorRT Documentation: Hardware Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#hardware-compat)
- [NVIDIA TensorRT Documentation: Compatibility Checks](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#compatibility-checks)
- [NVIDIA TensorRT Documentation: Plugin Shared Libraries](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugin-serialization)