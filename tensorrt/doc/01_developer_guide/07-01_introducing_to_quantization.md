# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introducing to Quantization](#introducing-to-quantization)
- [Quantization Workflows](#quantization-workflows)
- [Explicit Versus Implicit Quantization](#explicit-versus-implicit-quantization)
- [Quantization Scahems](#quantization-schemes)
- [Quantization Modes](#quantization-modes)
- [References](#references)

<br>

> [TensorRT 10.0] INT8 이외에 `FP8(E4M3)`, `signed INT4` quantization이 추가됨

# Introducing to Quantization

TensorRT는 양자화된 부동소수점 값을 표현하기 위해 더 낮은 정밀도의 타입을 지원한다. _Symmetric_ quantization을 지원하며, 양자화된 값은 signed INT8, FP8E4M3 (FP8 for short), 또는 signed INT4로 표현된다. 양자화된 값에서 역양자화된 값으로우 변환은 단순 곱셈이며, 

TensorRT의 quantization scheme은 INT8 및 FP8에 대해 weights뿐만 아니라 activation도 양자화한다. INT4의 경우에는 **weight-only-quantization**을 지원한다.

# Quantization Workflows

양자화된 네트워크를 생성하는 두 가지 방식이 있다.

- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)

_Post-training quantization_ (PTQ)는 네트워크를 학습한 후, scale factor를 도출한다. TensorRT에서는 _calibration_ 이라는 PTQ workflow를 제공한다. Calibration은 네트워크가 대표적인 입력 데이터에서 수행될 때, 각 activation 텐서 내에서 activation의 분포를 측정한다. 그러고, 해당 분포를 사용하여 텐서에 대한 scale 값을 측정한다.

_Quantization-aware training_ (QAT)는 학습하는 동안 scale factor를 계산한다. 이는 학습 과정에서 quantization 및 dequantization 연산의 영향을 보상하도록 한다.

TensorRT의 [Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)은 TensorRT에 의해서 최적화될 수 있는 QAT 모델을 생성하는데 도움을 주는 파이토치 라이브러리이다. 이 툴킷을 사용하여 파이토치에서 PTQ도 수행할 수 있으며, ONNX로 영자화된 모델을 추출할 수도 있다.

# Explicit Versus Implicit Quantization

양자화된 네트워크(quantized network)는 두 가지 방식으로 처리될 수 있으며, 두 방법은 상호 배타적이다.

- _Implicitly quantized_ network
- _Explicitly quantized_ network

두 방식의 주요 차이점은 명시적으로 quantization을 제어하는지, 또는 TensorRT 빌더가 어떤 연산과 텐서를 양자화하는지(implicit)이다. Implicit quantization은 오직 INT8 quantization에서만 지원되며 strong typing과 함께 지정될 수 없다 (타입이 auto-tuned되지 않으면서 activation을 INT8로 또는 INT8로부터 변환하는 유일한 방법은 Q/DQ operator를 통해서만 가능하기 때문이다). 

네트워크에 `QuantizeLayer`와 `DequantizeLayer` 레이어가 존재하면 TensorRT는 explicit quantization mode를 사용한다. 네트워크에 앞의 두 레이어가 없고 builder configuration에 INT8이 활성화되어 있다면, TensorRT는 implicit quantization mode를 사용한다. 오직 INT8에서만 implicit quantization mode를 지원한다.

_Implicitly quantized_ 네트워크에서 양자화가 될 후보인 각 activation tensor는 관련된 scale 값을 가지며 이는 calibration process 또는 `setDynamicRange` API 함수로 할당된다. TensorRT가 해당 텐서를 양자화하기로 결정했다면 이 scale 값을 사용하게 된다.

Implicitly quantized 네트워크를 처리할 때, TensorRT는 graph optimization을 적용 시 모델을 floating-point model로 취급한다. 그리고 layer execution time을 최적화할 수 있다면 INT8을 선택적으로 사용한다. 만약 INT8에서 해당 레이어가 더 빠르고 data input과 output에 대한 quantization scales을 가지고 있다면, INT8 정밀도의 커널이 해당 레이어에 할당된다. 그렇지 않다면, 더 높은 정밀도의 부동소수점(FP32, FP16, or BF16) 커널이 할당된다. 성능을 어느정도 희생하면서 정확성을 높이려면 `Layer::setOutputType`과 `Layer::setPrecision` API를 사용하여 높은 정밀도를 지정할 수 있다.

_Explicitly quantized_ 네트워크에서 quantized value와 unquantized value 간의 변환을 위한 scaling 연산은 그래프의 `IQuantizeLayer`와 `IDequantizeLayer` 노드로 명시적으로 표현된다. 이들을 Q/DQ 노드라고 부른다. Implicit quantization과는 달리, explicit quantization은 INT8 변환이 수행되는 위치를 명시적으로 지정하고, optimizer는 모델의 의미 체계에서 지시되는 precision conversion만 수행한다.

만약 INT8 이외의 변환도 추가한다면,
- 레이어의 정밀도가 증가할 수 있고 (예를 들어, INT8 구현 대신 FP16 커널 구현을 선택)
- 더 빠르게 실행되는 엔진을 얻을 수 있다 (예를 들어, float precision으로 지정된 레이어 실행에 INT8 커널 구현을 선택하거나, 그 반대의 경우)

ONNX는 explicitly quantized representation을 사용한다. 파이토치 또는 텐서플로우의 모델이 ONNX로 추출될 때, 각 프레임워크의 그래프에서 fake-quantization 연산은 Q/DQ로 추출된다. TensorRT는 이들 레이어의 의미 체계를 유지하기 떄문에 프레임워크에서 달성한 정확도에 근사하는 정확도를 기대할 수 있다. 최적화에서 quantization과 dequantization의 배치는 그대로 유지하지만, 부동소수점 연산의 순서는 모댈 내에서 바뀔 수 있기 때문에 그 결과가 bitwise로 일치하지는 않을 수 있다.

TensorRT의 PTQ와 달리, 프레임워크에서 QAT 또는 PTQ를 수행하고 ONNX로 추출하면 explicitly quantized model이 생성된다.

||Implicit Quantization|Explicit Quantization|
|--|--|--|
|Supported quantized data-types|INT8|INT8, FP8, INT4|
|User control over precision|Global builder flags and per-layer precision APIs.|Encoded directly in the model.|
|API|- Model + Scales (dynamic range API)<br>- Model + Calibration data|Model with Q/DQ layers.|
|Quantization scales|Weights:<br>- Set by TensorRT (internal)<br>-Range [-127, 128]<br>Activations:<br>- Set by calibration or specified by the user<br>- Range [-128, 127]|Weights and activations:<br>- Specified using Q/DQ ONNX operators<br>- INT8 Range [-128, 127]<br>- FP8 range: [-448, 448]<br>- INT4 range: [-8, 7] <br> Activations use per-tensor quantization.<br>Weights use either per-tensor quantization, per-channel quantization or block quantization.|

**참고 자료**

- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)
- [FP8 Formats for Deep Learning](https://arxiv.org/pdf/2209.05433.pdf)


# Quantization Schemes

## INT8

Scale $s$가 주어졌을 때, quantization 및 dequantization operators는 다음과 같이 표현할 수 있다.

- $x_q = \text{quantize}(x, s) := \text{roundWithTiesToEven}\left(\text{clip}\left(\frac{x}{s}, -128, 127\right)\right)$
  - $x$ is a high-precision floating point value to be quantized.
  - $x_q$ is quantied value in range [-128, 127].
  - `roundWithTiesToEven` is described [here](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).
- $x = \text{dequantize}(x_q, s) = x_q * s$

Explicit quantization에서는 모든 scale을 선택해야 한다. Implicit quantization mode에서는 사용자에 의해서 activation scale이 구성되거나 TensorRT의 calibration algorithms 중 하나를 사용하여 결정된다. Weight scale은 아래의 공식에 의해 계산된다 ([Post-Training Quantization Using Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c) 참조).


$$ s = \frac{\max{(\text{abs}{(x_{\min}^{\text{ch}}), \text{abs}(x_{\max}^{\text{ch}})})}}{127} $$

> 동일한 네트워크에서 FP8과 INT8를 함께 사용할 수 없다.

## FP8

- $x_q = \text{quantize}(x, s) := \text{roundWithTiesToEven}\left(\text{clip}\left(\frac{x}{s}, -448, 448\right)\right)$
  - $x$ is a high-precision floating point value to be quantized.
  - $x_q$ is quantied value in range [-448, 448].
  - $s$ is the quantization scale expressed using a 16-bit or 32-bit floating point.
- $x = \text{dequantize}(x_q, s) = x_q * s$

## INT4

- $x_q = \text{quantize}(x, s) := \text{roundWithTiesToEven}\left(\text{clip}\left(\frac{x}{s}, -8, 7\right)\right)$
  - $x$ is a high-precision floating point value to be quantized.
  - $x_q$ is quantied value in range [-8, 7].
  - $s$ is the quantization scale expressed using a 16-bit or 32-bit floating point.
- $x = \text{dequantize}(x_q, s) = x_q * s$

> 오직 weight-only quantization에서만 INT4를 지원한다.

# Quantization Modes

TensorRT는 아래의 3가지 quantization modes를 지원한다.

- **_Per-tensor quantization_** : 하나의 값이 전체 텐서를 스케일하는데 사용됨
- **_Per-channel quantization_** : 주어진 축에 따라 스케일 텐서가 브로드캐스트됨. 일반적으로 convolution neural network에서 channel 축을 사용.
- **_Block quantization_** : 단일 차원을 따라 고정된 크기의 1차원 blocks로 텐서를 나누며, scale factor는 각 블록마다 정의됨.

Quantization scale은 반드시 positive high-precision float(FP32, FP16, or BF16)으로 구성된다. Rounding method는 [round-to-nearest-ties-to-even](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even)이며 유효한 값으로 클램핑된다.

- INT8 : [-128, 127]
- FP8 : [-448, 448]
- INT4 : [-8, 7]

Explicit quantization에서 activation은 반드시 per-tensor quantization mode이어야 하며, weight는 모두 가능하다.

Implicit quantization에서는 weights가 engine optimization이 진행되는 중에 TensorRT에 의해서 양자화되며 per-channel quantization만 사용된다. Convolution, deconvolution, fully connected layers, 그리고 MatMul에 대해서 weights를 양자화한다.

Convolution에 per-channel quantization을 사용할 때, quantization axis는 반드시 output-channel axis이어야 한다. 예를 들어, weight를 `KCRS` 표기법으로 표현할 때, `K`가 output-channel axis이며 weight quantization 과정은 다음과 같이 표현할 수 있다.
```
For each k in K:singe
    For each c in C:
        For each r in R:
            For each s in S:
                output[k,c,r,s] := clamp(round(input[k,c,r,s] / scale[k]))
```

Quantization scale은 걔수(coefficients) 벡터이며, 반드시 quantization axis와 같은 크기를 갖는다. 

Dequantization은 아래와 같이 pointwise operation이라는 점만 제외하면 유사하게 수행된다.
```
output[k,c,r,s] := input[k,c,r,s] * scale[k]
```

## Block Quantization

Block quantization에서 요소들은 1-D blocks로 그룹화되며, 한 블록의 모든 요소들은 하나의 scale factor를 공유한다. 오직 INT4 2-D weight-only-quantization(WoQ)만 지원된다.

Scale tensor 차원은 blocking이 수행되는 하나의 차원(the blocking axis)을 제외하고 data tensor의 차원과 동일하다. 예를 들어, 2-D `RS` weights input과 blocking axis `R`, 그리고 block size `B`가 주어졌을 때, blocking axis의 scale은 block size에 따라 반복되며 아래와 같이 표현할 수 있다.
```
For each r in R:
    For each s in S:
        output[r,s] = clamp(round(input[r,s] / scale[r//B, s]))
```
위의 경우, scale은 2차원 배열이며 (R//B, S) 차원이다.

Dequantization 또한 비슷하게 수행된다.
```
output[r,s] = input[r,s] * scale[r//B, s]
```

# References

- [NVIDIA TensorRT Documentation: Introducing to Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization)
- [pytorch-quantization (github)](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [Paper: Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)