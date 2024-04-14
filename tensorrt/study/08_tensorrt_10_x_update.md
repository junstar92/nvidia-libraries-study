# Table of Contents

- [Table of Contents](#table-of-contents)
- [TensorRT 10.0.0 EA](#tensorrt-1000-early-access)
- [References](#references)

<br>

# TensorRT 10.0.0 Early Access

## Compatilbilities

- Minimum glibc version: 2.28 for Linux x86 build
  - This will be reverted for TensorRT 10.0 GA and will be compatible with glibc 2.17 (the minimum glibc version supported by TensorRT 8.6)
- Support for Python 3.6 and 3.7 has been dropped starting with TensorRT 10.0.
- Python 3.12 support has been added starting with TensorRT 10.0.

## Quantization

- INT4 Weight Only Quantization : Added support for weight compression using INT4 (hardware agnostic).
- Block Quantization

## Implementation & APIs

- Debug Tensors: 빌드 시, 텐서를 디버그하기 위한 mark tensor API 추가. 런타임 시, 텐서의 값이 쓰여질 때마다 user-defined 콜백 함수가 호출됨 (with the value, type, and dimensions).
- Runtime Allocation: `createExecutionContext` 호출 시, allocation strategy를 설정할 수 있음. User-managed allocation인 경우, `updateDeviceMemorySizeForShapes` API를 사용하여 실제 input shapes를 기반으로 필요한 크기를 쿼리할 수 있음
- Weight Streaming: 디바이스의 device memory보다 큰 strongly typed model을 실행할 수 있도록 해주는 기능.
- `kTACTIC_SHARED_MEMORY` flag: TensorRT 백엔드 CUDA 커널에 사용되는 shared memory budget control.
- V3 plugins 추가 ([Migrating V2 Plugins to IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#migrating-plugins) 참조)
- 다음의 classes가 추가됨
  - `nvinfer1::IPluginCreatorInterface`
- 다음의 enum values가 추가됨
  - `ExecutionContextAllocationStrategy::kSTATIC`
  - `ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE`
  - `ExecutionContextAllocationStrategy::kUSER_MANAGED`
  - `BuilderFlag::kWEIGHT_STREAMING`
  - `MemoryPoolType::kTACTIC_SHARED_MEMORY`

### `IExecutionContext` API 업데이트

- `IExecutionContext::updateDeviceMemorySizeForShapes()`

### `IPluginRegistry` API 업데이트

- `IPluginRegistry::registerCreator(IPluginCreatorInterface& creator, char const* const pluginNamespace)`
- `IPluginRegistry::deregisterCreator(IPluginCreatorInterface& creator)`

## Breaking API Changes

- TensorRT 9.3 및 이전 릴리즈에서 더 이상 사용되지 않는 API가 제거됨.
- 8.6, 9.X, 10.0 버전 간 호환성을 가짐.
- `TacticSource::kCUBLAS`와 `TacticSource::kCUDNN`이 기본적으로 비활성됨.

## Deprecated and Removed Features

- implicit batch support 제거
- Deprecated Classes
  - `IPluginV2DynamicExt`
  - `IPluginCreator`
  - `IPluginV2IOExt`
- Deprecated Functions
  - `IPluginRegistry::registerCreator(IPluginCreator&)`
  - `IPluginRegistry::deregisterCreator(IPluginCreator const&)`
  - `IPluginRegistry::getPluginCreator()`
  - `IPluginRegistry::getPluginCreatorList()`
- Deprecated Enums
  - `BuilderFlag::kWEIGHTLESS`
  - `TacticSource::kCUDNN`
  - `TacitcSource::kCUBLAS`
- Removed Functions
  - `IPluginCreator::getTensorRTVersion()`

# References

- [NVIDIA TensorRT Documentation: Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)