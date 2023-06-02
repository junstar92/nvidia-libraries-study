# Table of Contents

- [Table of Contents](#table-of-contents)
- [Network-Level Control of Precision](#network-level-control-of-precision)
- [Layer-Level Control of Precision](#layer-level-control-of-precision)
- [TF32](#tf32)
- [References](#references)

<br>

# Network-Level Control of Precision

기본적으로 TensorRT는 32-bit precision으로 동작하며, 16-bit floating point, 8-bit quantized floating point를 사용한 연산도 실행할 수 있다. 더 낮은 정밀도를 사용하면 더 적은 메모리 사용과 더 빠른 연산이 가능하다.

정밀도 감소의 지원 여부는 하드웨어에 따라 다르다 ([Hardware and Precision](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) 참조). 플랫폼에서 지원하는 정밀도를 확인하려면 builder를 통해 쿼리해야 한다.
```c++
if (builder->platformHasFastFp16()) { ... }
```

그리고, builder configuration의 플래그를 설정하여 TensorRT가 더 낮은 정밀도 구현을 선택하도록 알려줄 수 있다.
```c++
config->setFlag(BuilderFlag::kFP16);
```

세 가지 정밀도 플래그(`FP16`, `INT8`, `TF32`)가 있으며, 이들은 독립적으로 활성화될 수 있다. 참고로 전반적으로 실행시간이 느리거나 더 낮은 정밀도의 구현이 존재하지 않는다면, TensorRT는 설정한 것보다 더 높은 정밀도의 커널을 선택할 수 있다.

레이어의 정밀도를 선택할 때, TensorRT는 레이어를 실행할 때 필요한 weights를 자동으로 변환한다.

`FP16`와 `TF32` 정밀도를 사용하는 것은 비교적 간단하다. 하지만, `INT8` 정밀도를 사용할 때는 조금 복잡하다. INT8 정밀도를 사용하는 방법은 [Working with INT8](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)에서 자세히 설명한다.

> 주의해야 할 점은 비록 precision 플래그가 활성화되더라도, 엔진에 바인딩되는 input/output 텐서는 기본적으로 FP32이다. Input/output 텐서의 정밀도와 연산의 정밀도는 별도로 생각해야 하며, 텐서의 정밀도는 따로 설정해주어야 한다. 바인딩되는 input/output 텐서의 data type과 format을 설정하는 방법은 [I/O Formats](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors)에서 자세히 설명한다.

<br>

# Layer-Level Control of Precision

Builder 플래그로 설정하는 정밀도는 대략적인 제어를 제공한다. 그러나, 때때로 네트워크의 몇몇 부분은 더 높은 dynamic range를 필요로 하거나 수치적 정밀도에 민감할 수 있다. 이런 경우, 레이어마다 input과 output의 타입을 제한할 수 있다.
```c++
layer->setPrecision(DataType::kFP16);
```
이렇게 하면 input과 output에 대해 *선호하는 타입* 을 제공할 수 있다. 위의 예제 코드에서는 `DataType::kFP16`으로 설정했다.

또한, 레이어의 output에 대해 선호하는 타입을 설정할 수도 있다.
```c++
layer->setOutputType(out_tensor_index, DataType::kFLOAT);
```

위와 같이 설정하면 연산은 input에 대해 선호되는 것과 동일한 부동소수점 타입을 사용한다. 대부분의 TensorRT 구현은 input과 output에 대해 동일한 부동소수점 타입을 사용한다. 하지만, Convolution, Deconvolution, FullyConnected 레이어는 quantized INT8 input과 unquantized FP16 or FP32 output을 지원한다. 이는 때때로 정밀도를 유지하기 위해 quantized input으로부터 더 높은 정밀도의 출력이 필요하기 때문이다.

정밀도 제약을 설정하는 것은 TensorRT가 선호되는 타입의 input과 output에 대한 구현을 선택해야 한다고 **힌트**를 주며, 만약 이전 레이어의 output과 다음 레이어의 input의 타입이 일치하지 않는다면 reformat 연산을 추가한다. TensorRT는 builder configuration에 이러한 플래그를 사용하는 경우에만 해당 타입의 구현을 선택할 수 있다.

기본적으로 TensorRT는 설정된 정밀도의 구현이 성능이 더 좋은 경우에만 해당 정밀도의 구현을 선택한다. 만약 다른 구현이 더 빠르다면, TensorRT는 더 빠른 구현을 선택하고 경고를 출력한다. 이러한 동작은 builder configuration을 통해 타입 제약을 선호하도록 동작을 제어할 수 있다.
```c++
config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
```

만약 정밀도 제약이 선호된다면, TensorRT는 선호하는 정밀도 제약에 대한 구현이 없는 경우를 제외하고 해당 정밀도 구현을 선택한다. 선호하는 정밀도의 구현이 없다면, 경고를 출력하고 사용 가능한 가장 빠른 구현을 사용한다.

경고가 아닌 에러를 출력하고 싶다면, `PREFER` 대신 `OBEY`를 사용하면 된다.
```c++
config->setFlag(BuiderFlag::kOBEY_PRECISION_CONSTRAINTS);
```

> [sampleINT8API](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleINT8API) 샘플은 위에서 설명한 API들을 사용하여 정밀도를 낮추는 방법을 보여준다.

정밀도 제약은 *Optional* 이다. `layer->precisionIsSet()`을 통해 정밀도 제약이 선택되었는지 쿼리하여 알 수 있다. 만약 정밀도 제약이 설정되지 않았다면, `layer->getPrecision()`으로 반환되는 결과는 의미가 없다. Output 타입의 제약 또한 비슷하게 *Optional* 이다.

만약 `ILayer::setPrecision` 또는 `ILayer::setOutputType` API를 사용하여 설정된 제약이 없다면, `BuilderFlag::kPREFER_PRECISION_CONSTRAINTS` 또는 `BuilderFlag::kOBEY_PRECISION_CONSTRAINTS`는 무시된다. 각 레이어는 허용된 builder precision을 기반으로 모든 정밀도 또는 output 타입 중 자유롭게 선택할 수 있다.

`ITensor::setType()` API는 네트워크의 input/output 텐서 중 하나가 아니라면 텐서의 정밀도 제약을 설정하지 않는다. 또한, `layer->setOutputType()`과 `layer->getOutput(i)->setType()` 간에는 차이점이 있다. 전자는 레이어에서 TensorRT가 선택하는 구현을 제약하는 optional type이며, 후자는 네트워크의 input/output의 타입을 지정하며 네트워크의 input/output이 아니라면 무시된다 (단순히 제약이 아닌 input/output 텐서 자체의 타입이라고 이해하면 된다). 만약 두 API에서 설정한 타입이 다르다면(네트워크의 input/output이면서), TensorRT는 두 스펙을 모두 준수하도록 캐스트(cast)를 삽입한다. 그러므로 만약 네트워크의 output을 생성하는 레이어에서 `setOutputType()`을 호출한다면, 일반적으로 대응하는 네트워크 output을 동일한 타입으로 설정해야 한다.

<br>

# TF32

기본적으로 TensorRT는 TF32 Tensor Cores를 사용한다. Convolution 또는 행렬 곱셈과 같은 내적을 계산할 때, TF32는 아래의 내용들을 실행한다.

- Rounds the FP32 multiplications to FP16 precision but keeps the FP32 dynamic range.
- Computes an exact product of the rounded multiplicands.
- Accumulates the products in an FP32 sum.

TF32 텐서 코어를 사용하면 FP32를 사용하는 네트워크에서 일반적으로 정확도 손실없이 속도 향상을 얻을 수 있다. Weights나 activations의 HDR(high dynamic range)가 필요한 모델인 경우, TF32가 FP16보다 더 견고하다.

여기서 TF32 텐서 코어가 실제로 사용된다는 보장은 없다. 그리고 이를 사용하도록 강제할 방법도 없다. 언제든지 TensorRT는 TF32를 FP32로 fallback할 수 있고, TF32가 지원되지 않는 플랫폼에서는 항상 FP32로 fallback한다.

TF32를 비활성화하는 것은 가능하며, 이는 TF32 builder 플래그를 clear해주면 된다.
```c++
config->clearFlag(BuilderFlag::kTF32);
```

> `NVIDIA_TF32_OVERRIDE=0` 환경변수를 설정하면, `BuilderFlag::kTF32`를 설정해도 TF32의 사용을 비활성화한다. 이 환경변수를 0으로 설정하면 NVIDIA 라이브러리의 모든 defaults 또는 programmatic configuration을 오버라이드하기 때문에 TF32 텐서 코어로 FP32를 절대 가속하지 않는다 (이는 오직 디버깅용이다). 0 이외의 다른 값은 나중을 위해 예약되어 있다.

> `NVIDIA_TF32_OVERRIDE` 환경변수의 값을 다른 값으로 설정하는 것은 엔진을 실행할 때, unpredictable precision/performance 효과를 일으킬 수 있다. 따라서 엔진을 실행할 때 unset으로 두는 것이 베스트이다.

> 어플리케이션에서 TF32의 dynamic range보다 더 높은 dynamic range를 요구하지 않는 한, FP16의 성능이 항상 더 빠르므로 FP16이 더 좋은 솔루션이 될 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Reduced Precision](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision)
- [NVIDIA TensorRT Documentation: Hardware and Precision Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)
- [TensorRT Sample: sampleINT8API](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleINT8API)