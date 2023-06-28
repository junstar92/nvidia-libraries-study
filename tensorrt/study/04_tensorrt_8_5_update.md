# Table of Contents

- [Table of Contents](#table-of-contents)
- [TensorRT 8.5.1](#tensorrt-851)
  - [Architectures](#architectures)
  - [Layers](#layers)
  - [Plug-in](#plug-in)
  - [Implementation \& APIs](#implementation--apis)
  - [For DLA](#for-dla)
- [TensorRT 8.5.2](#tensorrt-852)
  - [Plug-in](#plug-in-1)
- [TensorRT 8.5.3](#tensorrt-853)
- [References](#references)

<br>

# TensorRT 8.5.1

TensorRT 8.5.1 버전이 릴리즈되면서 많은 내용들이 업데이트된 것으로 확인된다. 특히, 이전 버전까지는 execution context에서 `enqueueV2()`를 통해 추론을 수행했는데, 8.5에서는 `enqueueV3()`가 도입되면서 이와 관련된 API들이 많이 업데이트된 것으로 보인다.

이번 포스팅에서는 몇 가지 카테고리 별로 주요 업데이트 내용에 대해서 살펴본다.

## Architectures

- NVIDIA Hopper (H100) 아키텍처를 지원한다. H100의 텐서 코어를 사용하고, A100보다 향상된 MMA(matrix multiply-accumulate) 처리량을 제공하는 Compute capapbility 9.0의 딥러닝 커널을 지원한다 (for `FP32`, `TF32`, `FP16`, `INT8`).
  - 제공되는 커널은 `Asynchronous Transaction Barriers`, `Tensor Memory Accelerator`(TMA), `Thread Block Clusters`와 같은 H100의 새로운 기능을 활용하여 효율성을 향상시킨다.
- NVIDIA Ada lovelace 아키텍처를 지원한다. Compute capability 8.9용 `FP32`, `TF32`, `FP16`, `INT8` 딥러닝 커널을 지원한다.

## Layers

- 부울 타입을 지원하도록 다음의 레이어들이 업데이트되었다.
  - `IGatherLayer`, `ISliceLayer`, `IConstantLayer`, `IConcatenationLayer`
- 다음의 레이어들이 새로 추가되었다.
  - `INonZeroLayer`, `INMSLayer`(non-max suppression), `IOneHotLayer`, `IGridSampleLayer`

> 지원되는 레이어에 대한 정보는 [Operator's Reference](https://docs.nvidia.com/deeplearning/tensorrt/operators/index.html)에서 자세히 확인할 수 있다.

## Plug-in

- `RoiAlign` 플러그인이 새로 추가되었다. 이를 사용하여 ONNX의 `RoiAlign` Operator(opset-10 and opset-16)를 지원한다.

## Implementation & APIs

- 텐서의 shapes는 GPU에서 계산된 값에 따라 달라질 수 있다. 예를 들어, `INonZeroLayer`의 output 텐서에서 마지막 차원은 입력에서의 non-zero 값이 얼마나 존재하느냐에 따라 결정된다. 이에 대한 내용은 [Dynamically Shaped Output](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md#dynamically-shaped-output)에서 다루고 있다.
- Named input dimensions를 지원한다. ONNX 모델에서 같은 이름의 차원 파라미터는 동일하다고 간주된다 ([Named Dimensions](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md#named-dimensions) 참조). 이에 따라, `ITensor`에서 아래의 두 API가 업데이트되었다.
  - `ITensor::setDimensionName(int32_t, index, char const* name)`
  - `ITensor::getDimensionName(int32_t index)`
- **Heuristic-based builder tactic selection**을 지원한다. `IBuilderConfig::setFlag()`를 통해 `BuilderFlag::kENABLE_TACTIC_HEURISTIC` 플래그를 아래와 같이 설정해주면 된다.
  
  ```c++
  config->setFlag(BuilderFlag::kENABLE_TACFTIC_HEURISTIC);
  ```
- 바로 아래에서 설명할 preview features 지정을 위한 API가 추가되었다.
  - `IBuilderConfig::setPreviewFeature(PreviewFeature feature, bool enable)`
  - `IBuilderConfig::getPreviewFeature(PreviewFeature feature)`
- Core 라이브러리의 cuDNN 및 cuBLAS를 포함하여 외부 tactic sources를 비활성화하고 플러그인에서 cuDNN 및 cuBLAS 사용을 허용할 수 있다. 이는 `IBuilderConfig::setPreviewFeature()`를 통해 `PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805`를 설정해주면 된다.
- 새로운 preview feature인 `PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805`가 추가되었다. 이는 빌드 시간을 줄이는 것이 목적이다.
- Lazy Module Loading을 지원한다. CUDA 기능이며 GPU 메모리 소모를 상당히 감소시킬 수 있다 (CUDA 문서의 [Lazy Loading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)과 TensorRT 문서의 [CUDA Lazy Loading](/tensorrt/doc/01_developer_guide/05_how_tensorrt_works.md#cuda-lazy-loading) 참조).
- CUDA 기능인 L2 Persistent cache를 지원한다 ([L2 Persistent Cache Management](/tensorrt/doc/01_developer_guide/05_how_tensorrt_works.md#l2-persistent-cache-management) 참조).
- `IResizeLayer` API 추가
  - `IResizeLayer::setCubicCoeff()`
  - `IResizeLayer::getCubicCoeff()`
- 새로운 enum value 추가
  - `InterpolationMode::kCUBIC`
  - `FillOperation::kRANDOM_NORMAL`
  - `BuilderFlag::kREJECT_EMPTY_ALGORITHMS`
  - `BuilderFlag::kENABLE_TACTIC_HEURISTIC`
  - `TacticSource::kJIT_CONVOLUTIONS`
  - `DataType::kUINT8`
- 새로운 enum class 추가
  - `TensorIOMode`
  - `PreviewFeature`

### `ICudaEngin` API 업데이트

Binding index로 텐서의 정보를 쿼리하는 API들은 대부분 deprecated 되었으며, 주로 텐서의 이름으로 관련된 정보를 쿼리하는 API가 추가됨
- `ICudaEngine::getTensorShape(char const *tensorName)`
- `ICudaEngine::getTensorDataType(char const *tensorName)`
- `ICudaEngine::getTensorLocation(char const *tensorName)`
- `ICudaEngine::isShapeInferenceIO(char const *tensorName)`
- `ICudaEngine::getTensorIOMode(char const *tensorName)`
- `ICudaEngine::getTensorBytesPerComponent(char const *tensorName[, int32_t profileIndex])`
- `ICudaEngine::getTensorFormat(char const *tensorName[, int32_t profileIndex])`
- `ICudaEngine::getTensorFormatDesc(char const *tensorName[, int32_t profileIndex])`
- `ICudaEngine::getProfileShape(char const *tensorName, int32_t profileIndex, OptProfileSelector select)`
- `ICudaEngine::getNbIOTensors()`
- `ICudaEngine::getIOTensorName(int32_t index)`

### `IExecutionContext` API 업데이트

`ICudaEngine`과 마찬가지로 binding index가 아닌 텐서의 이름으로 관련 정보를 쿼리 및 설정하는 API가 주로 추가되었으며, 또한, `enqueueV3()`와 관련된 API들도 추가됨
- `IExecutionContext::getTensorStrides(char const *tensorName)`
- `IExecutionContext::setInputShape(char const *tensorName, Dims const& dims)`
- `IExecutionContext::getTensorShape(char const *tensorName)`
- `IExecutionContext::setTensorAddress(char const *tensorName, void *data)`
- `IExecutionContext::getTensorAddress(char const *tensorName)`
- `IExecutionContext::setInputTensorAddress(char const *tensorName, void const *data)`
- `IExecutionContext::getOutputTensorAddress(char const *tensorName)`
- `IExecutionContext::setOutputAllocator(char const *tensorName, IOutputAllocator *outputAllocator)`
- `IExecutionContext::getOutputAllocator(char const *tensorName)`
- `IExecutionContext::getMaxOutputSize(char const *tensorName)`
- `IExecutionContext::setTemporaryStorageAllocator(IGpuAllocator *allocator)`
- `IExecutionContext::getTemporaryStorageAllocator()`
- `IExecutionContext::enqueueV3(cudaStream_t stream)`
- `IExecutionContext::setPersistentCacheLimit(size_t size)`
- `IExecutionContext::getPersistentCacheLimit()`
- `IExecutionContext::setNvtxVerbosity(ProfilingVerbosity verbosity)`
- `IExecutionContext::getNvtxVerbosity()`

### `INetworkDefinition` API 업데이트

새로 추가된 네트워크와 관련된 API가 추가됨

- `INetworkDefinition::addOneHot()`
- `INetworkDefinition::addNonZero()`
- `INetworkDefinition::addGridSample()`
- `INetworkDefinition::addNMS()`

## For DLA

- `IShuffleLayer`를 DLA로 오프로딩하는 것을 지원한다. DLA에서 `IShuffleLayer` 실행하는 것에 대한 제약사항은 [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)를 참조.

> Limitations 및 Fixed Issues는 공식 문서의 [Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)를 참조 바람.

# TensorRT 8.5.2

## Plug-in

- Stable Diffusion 데모를 위한 플러그인들이 추가됨
  - fused Multihead Self-Attention
  - fused Multihead Cross-Attention
  - Layer Normalization
  - Group Normalization
  - targeted fusions (such as Split+GeLU)

# TensorRT 8.5.3

중요한 업데이트는 따로 없음

<br>

# References

- [NVIDIA TensorRT Documentation: Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [NVIDIA CUDA Documentation: Lazy Loading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)
- [TensorRT API Documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)