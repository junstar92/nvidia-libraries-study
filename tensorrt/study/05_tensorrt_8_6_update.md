# Table of Contents

- [Table of Contents](#table-of-contents)
- [TensorRT 8.6.1](#tensorrt-861)
  - [Compatibilities](#compatibilities)
  - [Layers](#layers)
  - [Libraries](#libraries)
  - [Implementation \& APIs](#implementation--apis)
  - [Samples](#samples)
  - [Deprecated Functions/Enums/Macros](#deprecated-functionsenumsmacros)
- [References](#references)

<br>

# TensorRT 8.6.1

## Compatibilities

- CUDA 12.x 버전이 TensorRT 8.6부터 지원된다. CUDA 11.x 빌드는 CUDA 12.x로 컴파일된 라이브러리 또는 어플리케이션과 호환되지 않으며 unexpected behavior를 유발할 수 있다.
- Hardware Compatibility에 대한 지원이 추가되었다. 이를 통해 **한 GPU 아키텍처에서 빌드된 엔진이 다른 아키텍처의 GPU에서 동작**할 수 있다. 이는 NVIDIA Ampere 및 그 이후의 아키텍처에서만 지원된다. 이를 사용하면 latency 및 throughput이 저하될 수 있다.
- 기본적으로 TensorRT 엔진은 오직 빌드된 TensorRT 버전에서만 호환된다. 하지만 TensorRT 8.6부터, 적절한 build configuration을 통해 major version은 같고, minor version이 다른 TensorRT 버전과 호환되는 엔진을 빌드할 수 있다. 이에 대한 정보는 [Version Compatibility](/tensorrt/doc/01_developer_guide/06-01_about_compatibility.md#version-compatibility)에서 확인할 수 있다.

## Layers

- ONNX의 `ReverseSequence` 연산자를 지원하기 위해 `IReverseSequenceLayer`가 추가되었다.
- ONNX의 `InstanceNormalization`, `GroupNormalization`, `LayerNormalization` 연산자를 지원하기 위해 `INormalizationLayer`가 추가되었다.
- `nvinfer1::ICastLayer` 인터페이스가 도입되었으며, 이는 입력 텐서의 데이터 타입 변환을 지원한다 (`FP32`, `FP16`, `INT32`, `INT8`, `UINT8`, `BOOL` 간의 변환0). ONNX parser는 cast를 구현하기 위해 `IIdentityLayer` 대신 `ICastLayer`를 사용하도록 업데이트되었다.

## Libraries

- 배포 시, memory consumption의 우선순위가 높은 경우, restricted runtime installation options (**lean or dispatch runtime mode**)를 도입했다.
  - `Lean Runtime Installation`: Full installation보다 훨씬 작으며 version compatible builder flag로 빌드된 엔진을 로드하고 실행할 수 있다. 이 라아브러리는 plan 파일을 빌드하는 기능을 제공하지 않는다.
  - `Dispath Runtime Installation`: 이를 통해 minimum memory consumption으로 배포할 수 있고, version compatible builder flag로 빌드된 엔진을 로드 및 실행할 수 있고 lean runtime을 포함할 수 있다. 마찬가지로 이 라이브러리에서 plan을 빌드하는 기능을 제공하지 않는다.

  이에 대한 내용은 [Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)에서도 언급하고 있다.

## Implementation & APIs

- 다음의 성능 향상이 업데이트되었다.
  - Multi-Head Attention (MHA) fusions 향상됨 (트랜스포머와 같은 네트워크의 속도가 향상됨).
  - Engine build time과 dynamic shapes를 가진 Transformer-based networks의 성능이 향상됨.
  - LSTM 또는 Transformer-like networks를 실행할 때 `enqueueV2()`와 `enqueueV3()`에서의 불필요한 `cuStreamSynchronize()` 호출을 피한다.
  - NVIDIA Hopper GPUs의 다양한 네트워크의 성능이 향상됨.
- Optimization level builder flag가 추가되었다. 이는 TensorRT가 더 성능이 좋은 tactics를 찾도록 엔진 빌드 시간을 더 사용하도록 하거나 탐색 범위를 줄여서 엔진을 더 빠르게 빌드하도록 할 수 있다. 이에 대한 사용법은 [Builder Optimization Level](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt-builder-optimization-level)을 참조.
- Multi-streams API가 추가되었다. 이를 통해 TensorRT가 스트림을 사용하여 병렬로 수행될 수 있는 부분을 병렬로 실행할 수 있다. 잠재적으로 더 좋은 성능을 보여줄 수 있으며, 자세한 내용은 [Within-Inference Multi-Streaming](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#within-inference-multi-streaming)에서 다루고 있다.
- 다음의 새로운 클래스들이 추가됨
  - `IReverseSequenceLayer`
  - `INormalizationLayer`
  - `ILoggerFinder`
- 다음의 매크로들이 추가됨
  - `NV_TENSORRT_RELEASE_TYPE`
  - `NV_TENSORRT_RELEASE_TYPE_EARLY_ACCESS`
  - `NV_TENSORRT_RELEASE_TYPE_RELEASE_CANDIDATE`
  - `NV_TENSORRT_RELEASE_TYPE_GENERAL_AVAILABILITY`
- `getBuilderSafePluginRegistry()` 함수가 추가됨
- 다음의 enum class가 추가됨
  - `HardwareCompatibilityLevel`
  - `TempfileControlFlag`
- 다음의 enum values가 추가됨
  - `BuilderFLag::kVERSION_COMPATIBLE`
  - `BuilderFlag::kEXCLUDE_LEAN_RUNTIME`
  - `DataType::kFP8`
  - `LayerType::kREVERSE_SEQUENCE`
  - `LayerType::kNORMALIZATION`
  - `LayerType::kCAST`
  - `MemoryPoolType::kTACTIC_DRAM`
  - `PreviewFeature::kPROFILE_SHARING_0806`
  - `TensorFormat::kDHWC`
  - `UnaryOperation::kISINF`

### `IAlgorithmIOInfo` API 업데이트

- `IAlgorithmIOInfo::getVectorizedDim()`
- `IAlgorithmIOInfo::getComponentsPerElement()`

### `IBuilderConfig` API 업데이트

- `IBuilderConfig::setBuilderOptimizationLevel(int32_t level)` : 설정 가능한 level은 0부터 5 사이의 값이며, 기본값은 3이다.
- `IBuilderConfig::getBuilderOptimizationLevel()`
- `IBuilderConfig::setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel)`
- `IBuilderConfig::getHardwareCompatibilityLevel()`
- `IBuilderConfig::setPluginsToSerialize(char const* const* paths, int32_t nbPaths)` : Version-compatible 엔진과 함께 직렬화될 플러그인 라이브러리를 설정한다.
- `IBuilderConfig::getPluginToSerialize(int32_t index)`
- `IBuilderConfig::getNbPluginsToSerialize()`
- `IBuilderConfig::getMaxAuxStreams()`
- `IBuilderConfig::setMaxAuxStreams(int32_t nbStreams)`

### `IBuilder` API 업데이트

- `IBuilder::getPluginRegistry()`

### `ICudaEngine` API 업데이트

- `ICudaEngine::getHardwareCompatibilityLevel()`
- `ICudaEngine::getNbAuxStreams()`

### `IExecutionContext` API 업데이트

- `IExecutionContext::setAuxStreams(cudaStream_t* auxStreams, int32_t nbStreams)`

### `ILayer` API 업데이트

- `ILayer::setMetadata(char const* metadata)`
- `ILayer::getMetadata()`

### `INetworkDefinition` API 업데이트

- `INetworkDefinition::addCast()`
- `INetworkDefinition::addNormalization()`
- `INetworkDefinition::addReverseSequence()`

### `IPluginRegistry` API 업데이트

- `IPluginRegistry::isParentSearchEnabled()`
- `IPluginRegistry::setParentSearchEnabled(bool const enabled)`
- `IPluginRegistry::loadLibrary(AsciiChar const* pluginPath)`
- `IPluginRegistry::deregisterLibrary(PluginLibraryHandle handle)`

### `IRuntime` API 업데이트

- `IRuntime::setTemporaryDirectory(char const* path)`
- `IRuntime::getTemporaryDirectory()`
- `IRuntime::setTempfileControlFlags(TempfileControlFlags flags)`
- `IRuntime::getTempfileControlFlags()`
- `IRuntime::getPluginRegistry()`
- `IRuntime::setPluginRegistryParent()` (API 문서에서 검색되지 않음)
- `IRuntime::loadRuntime(char const* path)`
- `IRuntime::setEngineHostCodeAllowed(bool allowed)`
- `IRuntime::getEngineHostCodeAllowed()`

### `ITopKLayer` API 업데이트

- `ITopKLayer::setInput(int32_t index, ITensor& tensor)`

## Samples

- [onnx_custom_plugin](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_custom_plugin) 샘플이 추가되었다. 이 샘플에서는 커스텀 또는 지원되지 않는 레이어가 있는 ONNX 모델을 파싱할 때, 어떻게 플러그인을 사용하는지에 대한 방법을 설명한다. 단계별로 자세하게 설명하고 있다.

## Deprecated Functions/Enums/Macros

- Deprecated Functions
  - `FieldMap::FieldMap()`
  - `IAlgorithmIOInfo::getTensorFormat()`
- Deprecated Enums
  - `PreviewFeature::kFASTER_DYNAMIC_SHAPES`
- Deprecated Macros
  - `NV_TENSORRT_SONAME_MAJOR`
  - `NV_TENSORRT_SONAME_MINOR`
  - `NV_TENSORRT_SONAME_PATCH`


<br>

# References

- [NVIDIA TensorRT Documentation: Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [TensorRT API Documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)
- [TensorRT Python Sample: onnx_custom_plugin](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_custom_plugin)