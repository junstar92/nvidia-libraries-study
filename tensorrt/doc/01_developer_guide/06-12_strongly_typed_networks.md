# Table of Contents

- [Table of Contents](#table-of-contents)
- [Strongly Typed Networks](#strongly-typed-networks)
- [References](#references)

<br>

> Updated in TensorRT 10.0.0

# Strongly Typed Networks

## Network-Level Control of Precision

기본적으로 TensorRT는 텐서 타입을 자동으로 조정하여 가장 빠른 엔진을 생성한다. 이로 인하여 TensorRT가 선택한 것보다 더 높은 정밀도로 레이어를 실행해야 하는 경우에는 정확도가 하락할 수 있다. 한 가지 방법은 `ILayer::setPrecision`과 `ILayer::setOutputType` API를 사용하여 레이어의 I/O 타입과 실행 정밀도를 제어하는 것이다. 이는 효과적이지만 최고의 정확도를 얻기 위해 어떤 레이어를 높은 정밀도로 실행해야 하는지 파악하는 것이 어려울 수 있다.

다른 대안은 Automatic mixed precision training 혹은 Quantization aware training 등을 사용하여 모델 자체에서 낮은 정밀도를 사용하도록 하고, TensorRT가 이 정밀도를 따르도록 하는 것이다.

TensorRT에 네트워크가 strongly typed하도록 지정할 때, 각 중간 텐서 및 결과 텐서는 [operator type specification](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)의 규칙을 따라 타입을 추론한다. 타입이 자동 조정되지 않기 때문에 strongly typed network로 빌드된 엔진은 TensorRT가 텐서 타입을 선택하는 엔진보다 느릴 수 있다. 반면, 평가하는 커널 수가 적으므로 빌드 시간은 향상될 수 있다.

> Strongly typed networks는 DLA에서 지원되지 않는다.

Strongly typed network는 다음과 같이 생성할 수 있다.

```c++
IBuilder* builder = ...;
INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
```

Strongly typed networks에서는 레이어의 API인 `setPrecision`과 `setOutputType`이 허용되지 않으며, `kFP16`, `kBF16`, `kFP8`, `kINT8` builder precision 플래그도 허용되지 않는다. `kTF32` 플래그는 허용된다.

# References

- [NVIDIA TensorRT Documentation: Strongly Typed Networks](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#strongly-typed-networks)