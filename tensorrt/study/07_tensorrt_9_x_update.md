# Table of Contents

- [Table of Contents](#table-of-contents)
- [TensorRT 9.3.0](#tensorrt-930)
- [TensorRT 9.2.0](#tensorrt-920)
- [TensorRT 9.1.0](#tensorrt-910)
- [TensorRT 9.0.1](#tensorrt-901)
- [TensorRT 9.0.0 EA](#tensorrt-900-early-access)
- [References](#references)

<br>

> TensorRT 9는 공식적으로 릴리즈되지 않고 문서도 없었는데, TensorRT 10.0이 릴리즈되면서 릴리즈 노트가 없데이트된 것으로 보인다. 구체적인 업데이트 내용보다 릴리즈 노트를 통해 TensorRT 9.x.x에서 업데이트된 내용들을 살펴보았다.

> TensorRT 9 버전 이후는 대부분 LLM을 위한 기능들이 추가된 것으로 보인다. `FP8`, `bfloat16`, `INT64` data type 추가와 API 업데이트 이외에 특별한 업데이트는 없는 듯함.

# TensorRT 9.3.0

## Implementation & APIs

- `getDeviceMemorySizeForProfile` API가 추가됨.
- LLM의 builder speed가 향상됨.

# TensorRT 9.2.0

## Implementation & APIs

- `FP16` inputs/outputs format에 대한 `FP32` accumulation 지원 추가
- 다음의 새로운 클래스가 추가됨
  - `ISerializeConfig`
- 다음의 새로운 enum value가 추가됨
  - `SerializationFlag::kEXCLUDE_WEIGHTS`
  - `SerializationFlag::kEXCLUDE_LEAN_RUNTIME`
  - `BuilderFlag::kWEIGHTLESS`

### `ICudaEngine` API 업데이트

- `ICudaEngine::createSerializationConfig()`
- `ICudaEngine::serializeWithConfig(ISerializationConfig& config)`

# TensorRT 9.1.0

> 거의 대부분 9.0.1 버전과 동일

## Implementation & APIs

- 다음의 새로운 enum value가 추가됨
  - `nvonnxparser::ErrorCode::kUNSUPPORTED_NODE_ATTR`
  - `nvonnxparser::ErrorCode::kUNSUPPORTED_NODE_INPUT`
  - `nvonnxparser::ErrorCode::kUNSUPPORTED_NODE_DATATYPE`
  - `nvonnxparser::ErrorCode::kUNSUPPORTED_NODE_DYNAMIC`
  - `nvonnxparser::ErrorCode::kUNSUPPORTED_NODE_SHAPE`

### `IRefitter` API 업데이트

- `IRefitter::getNamedWeights(char const* weightsName)`
- `IRefitter::getWeightsLocation(char const* weightsName)`
- `IRefitter::unsetNamedWeights(char const* weightsName)`
- `IRefitter::setWeightsValidation(bool weightsValidation)`
- `IRefitter::getWeightsValidation()`
- `IRefitter::refitCudaEngineAsync(cudaStream_t stream)`

### `IParserError` API 업데이트

- `nvonnxparesr::IParserError::nodeName()`
- `nvonnxparser::IParserError::nodeOperator()`

# TensorRT 9.0.1

> 거의 대부분 9.0.0 버전과 동일

## Compatilbilities

- NVIDIA GH200 Grace Hopper Superchip 지원.

## Limitations

- `QuantizeLinear`와 `DequantizeLinear`는 ONNX opset 19를 사용하더라도 오직 FP32 scale and date를 지원한다. 만약 입력이 FP32가 아니라면 `QuantizeLayer`의 입력과 `DequantizeLayer` output에 `Cast`를 사용하여 FP32로 변환해야 한다.
- 몇몇 엔진에 대해 `EngineInspector::getLayerInformation`는 불완전한 JSON data를 반환할 수 있다.

# TensorRT 9.0.0 Early Access

## Compatilbilities

- at least NVIDIA driver: r450(on Linux) or r452(on Windows)
- minimum CUDA version: 11.0

## Supported Data Type

- Added support for `bfloat16` data types on NVIDIA Ampere GPUs and newer architectures.
- Added support for `E4M3 FP8` data type on NVIDIA Hopper GPUs using explicit quantization (this allows utilizing TensorRT with `TransformerEngine`(https://github.com/NVIDIA/TransformerEngine) based FP8 models).
- Added `bfloat16` and FP8 I/O datatypes in plugins.
- Added support of networks running in INT8 precision where **Smooth Quantization** is used.
- Added support for `INT64` data type. The ONNX parser no longer automatically casts INT64 to INT32.

## ONNX

- Added support for ONNX local functions when parsing ONNX models with the ONNX parser.

## Implementation & APIs

- Provides enhanced support for Large Language Models, including the following networks: `GPT`, `GPT-J`, `GPT-Neo`, `GPT-NeoX`, `BART`, `Bloom`, `Bloomz`, `OPT`, `T5`, `FLAN-T5`
- caching JIT-compiled code 지원이 추가됨. 이는 `BuilderFlag::DISABLE_COMPILATION_CACHE` 플래그를 설정하여 비활성화할 수 있다. Comilation cache는 timing cache의 일부이며, 기본적으로 timing cache의 일부로 직렬화되어 cache size를 상당히 증가시킬 수 있다.
- `IEngineInspector`가 LSTMs와 Transformers 네트워크에 대한 더 정확한 레이어 정보를 출력함.
- 다음의 새로운 enum value가 추가됨
  - `IBuilderFlag::kFP8`
  - `IBuilderFlag::kERROR_ON_TIMING_CACHE_MISS`
  - `IBuilderFlag::kBF16`
  - `IBuilderFlag::kDISABLE_COMPILATION_CACHE`
  - `DataType::kBF16`
  - `DataType::kINT64`
  - `NetworkDefinitionCreationFlag::kSTRONGLY_TYPES`
- 다음의 새로운 클래스가 추가됨
  - `IProgressMonitor`

### `INetworkDefinition` API 업데이트

- `INetworkDefinition::addFill(Dims dimensions, FillOperation op, DataType outputType)`
- `INetworkDefinition::IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale, DataType outputType)`

### `IBuilder` API 업데이트

- `IBuilder::setProgressMonitor(IProgressMonitor* monitor)`
- `IBuilder::getProgressMonitor()`

### `IFillLayer` API 업데이트

- `IFillLayer::setAlphaInt64(int64_t alpha)`
- `IFillLayer::getAlphaInt64()`
- `IFillLayer::setBetaInt64(int64_t beta)`
- `IFillLayer::getBetaInt64()`
- `IFillLayer::isAlphaBetaInt64()`
- `IFillLayer::getToType()`
- `IFillLayer::setToType(DataType toType)`

### `IQuantizeLayer` API 업데이트

- `IQuantizeLayer::getToType()`
- `IQuantizeLayer::setToType(DataType toType)`

### `IDequantizeLayer` API 업데이트

- `IDequantizeLayer::getToType()`
- `IDequantizeLayer::setToType(DataType toType)`

### `INetworkDefinition` API 업데이트

- `INetworkDefinition::getFlags()`
- `INetworkDefinition::getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag)`

## Breaking API Changes

- `ICaffeParser`, `IUffParser` 클래스와 관련된 클래스 및 API가 제거됨.
- `libnvparsers` 라이브러리 제거됨.

## Deprecated and Removed Features

- 이번 릴리즈부터 Ubuntu 18.04를 지원하지 않음
- Deprecated Plugins
  - `BatchedNMS_TRT`
  - `BatchedNMSDynamic_TRT`
  - `BatchTilePlugin_TRT`
  - `Clip_TRT`
  - `CoordConvAC`
  - `CropAndResize`
  - `EfficientNMS_ONNX_TRT`
  - `CustomGeluPluginDynamic`
  - `LReLU_TRT`
  - `NMSDynamic_TRT`
  - `NMS_TRT`
  - `Normalize_TRT`
  - `Proposal`
  - `SingleStepLSTMPlugin`
  - `SpecialSlice_TRT`
  - `Split`
- Deprecated Classes
  - `NvUtils`
- Deprecated Functions
  - `nvinfer1::INetworkDefinition::addFill(nvinfer1::Dims dimensions, nvinfer1::FillOperation op)`
  - `nvinfer1::INetworkDefinition::addDequantize(nvinfer1::ITensor &input, nvinfer1::ITensor& scale)`
  - `nvinfer1::INetworkDefinition::addQuantize(nvinfer1::ITensor &input, nvinfer1::ITensor& scale)`
- Deprecated Enums
  - `nvinfer1::TacticSource::kCUBLAS_LT`
  - `nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM`

# References

- [NVIDIA TensorRT Documentation: Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)