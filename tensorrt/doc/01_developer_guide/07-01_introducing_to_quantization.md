# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introducing to Quantization](#introducing-to-quantization)
- [Quantization Workflows](#quantization-workflows)
- [Explicit Versus Implicit Quantization](#explicit-versus-implicit-quantization)
- [Per-Tensor and Per-Channel Quantization](#per-tensor-and-per-channel-quantization)
- [References](#references)

<br>

# Introducing to Quantization

TensorRT는 quantized floating point 값을 표현하기 위해 8-bit integers를 지원한다. Quantization(양자화) scheme은 _symmetric uniform_ quantization이며, 양자화된(quantized) 값은 부호가 있는 INT8로 표현되고 양자화된 값을 다시 원래의 값으로 변환은 단순한 곱셈이다. 역방향에서 quantization은 reciprocal scale과 roudning/clamping을 사용한다.

Quantization scheme은 weights 뿐만 아니라 activations의 양자화도 포함한다.

Activations에 대한 양자화 방식은 특정 데이터에 대해 rounding error와 precision error와의 균형을 가장 잘 맞추는 scale을 찾는 calibration 알고리즘에 따라 다르다. TensorRT에서는 몇 가지 caliration 방법을 지원하며, 이에 대한 내용은 [Post-Training Quantization Using Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c)에서 찾아볼 수 있다.

Weights에 대한 양자화는 아래의 공식을 따른다.

$$ s = \frac{\max{(\text{abs}{(x_{\min}), \text{abs}(x_{\max})})}}{127} $$

여기서 $x_{\min}$ 과 $x_{\max}$ 는 weights 텐서의 floating point minimum과 maximum 값이다.

주어진 스케일에서 quantize/dequantize 연산은 다음과 같이 표현할 수 있다.

- $x_q = \text{quantize}(x, s) := \text{roundWithTiesToEven}\left(\text{clip}\left(\frac{x}{s}, -128, 127\right)\right)$
  - $x_q$ is quantied value in range [-128, 127].
  - $x$ is a floating point value of the activation.
  - `roundWithTiesToEven` is described [here](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).
- $x = \text{dequantize}(x_q, s) = x_q * s$

Orin device의 DLA에서의 양자화 방식은 `roundWithTiesToNearestEven`을 사용하여 [link](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)에서 자세히 살펴볼 수 있다.

> 모든 quantized operations를 사용하려면, builder configuration에서 INT8 플래그를 설정해주어야 한다.

<br>

# Quantization Workflows

양자화된 네트워크를 생성하는 두 가지 방식이 있다.

- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)

_Post-training quantization_ (PTQ)는 네트워크를 학습한 후, scale factor를 도출한다. TensorRT에서는 _calibration_ 이라는 PTQ workflow를 제공한다. Calibration은 네트워크가 대표적인 입력 데이터에서 수행될 때, 각 activation 텐서 내에서 activation의 분포를 측정한다. 그러고, 해당 분포를 사용하여 텐서에 대한 scale 값을 측정한다.

_Quantization-aware training_ (QAT)는 학습하는 동안 scale factor를 계산한다. 이는 학습 과정에서 quantization 및 dequantization 연산의 영향을 보상하도록 한다.

TensorRT의 [Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)은 TensorRT에 의해서 최적화될 수 있는 QAT 모델을 생성하는데 도움을 주는 파이토치 라이브러리이다. 이 툴킷을 사용하여 파이토치에서 PTQ도 수행할 수 있으며, ONNX로 영자화된 모델을 추출할 수도 있다.

<br>

# Explicit Versus Implicit Quantization

양자화된 네트워크(quantized network)는 두 가지 방식으로 표현될 수 있다.

- _Implicitly quantized_ network
- _Explicitly quantized_ network

Implicit quantized network에서 양자화된 각 텐서는 해당 텐서에 연관된 스케일을 가지고 있다. 텐서를 읽거나 쓸 때, 스케일은 값을 암시적으로 양자화/역양자화할 때 사용된다.

Implicitly quantized network를 처리할 때, TensorRT는 그래프 연산을 적용에서 네트워크 모델을 floating-point model로 취급하고, 실행 시간에 기회에 따라서 INT8을 사용하여 레이어를 최적화한다. 만약 레이어가 INT8에서 속도가 더 빠르면, INT8로 실행한다. 더 느리다면, FP32 또는 FP16을 사용한다. 이 모드에서는 오직 성능으로 최적화하며, 사용자는 INT8이 사용되는 위치를 거의 제어할 수 없다. 심지어 API 레벨에서 레이어의 정밀도를 명시적으로 설정하더라도, TensorRT는 그래프 최적화 중에 다른 레이어와 융합(fusion)할 수 있고 INT8로 연산하라는 정보를 잃을 수 있다. TensorRT의 **PTQ**는 implicitly quantized network를 생성한다.

Explicitly quantized network에서 quantized value와 unquantized value 간의 변환을 위한 scaling 연산은 그래프의 `IQuantizeLayer`와 `IDequantizeLayer` 노드로 명시적으로 표현된다. 이들을 Q/DQ 노드라고 부른다. Implicit quantization과는 달리, explicit quantization은 INT8 변환이 수행되는 위치를 명시적으로 지정하고, optimizer는 모델의 의미 체계에서 지시되는 precision conversion만 수행한다.

만약 INT8 이외의 변환도 추가한다면,
- 레이어의 정밀도가 증가할 수 있고 (예를 들어, INT8 구현 대신 FP16 커널 구현을 선택)
- 더 빠르게 실행되는 엔진을 얻을 수 있다 (예를 들어, float precision으로 지정된 레이어 실행에 INT8 커널 구현을 선택하거나, 그 반대의 경우)

ONNX는 explicitly quantized representation을 사용한다. 파이토치 또는 텐서플로우의 모델이 ONNX로 추출될 때, 각 프레임워크의 그래프에서 fake-quantization 연산은 Q/DQ로 추출된다. TensorRT는 이들 레이어의 의미 체계를 유지하기 떄문에 프레임워크에서 달성한 정확도에 근사하는 정확도를 기대할 수 있다. 최적화에서 quantization과 dequantization의 배치는 그대로 유지하지만, 부동소수점 연산의 순서는 모댈 내에서 바뀔 수 있기 때문에 그 결과가 bitwise로 일치하지는 않을 수 있다.

TensorRT의 PTQ와 달리, 프레임워크에서 QAT 또는 PTQ를 수행하고 ONNX로 추출하면 explicitly quantized model이 생성된다.

||Implicit Quantization|Explicit Quantization|
|--|--|--|
|User control over precision|Little control: INT8 is used in all kernels for which it accelerates performance.|Full control over quantization/dequantization boundaries.|
|Optimization criterion|Optimize for performance.|Optimize for performance while maintaining arithmetic precision (accuracy).|
|API|- Model + Scales (dynamic range API)<br>- Model + Calibration data|Model with Q/DQ layers.|
|Quantization scales|Weights:<br>- Set by TensorRT (internal)<br>-Range [-127, 128]<br>Activations:<br>- Set by calibration or specified by the user<br>- Range [-128, 127]|Weights and activations:<br>- Specified using Q/DQ ONNX operators<br>- Range [-128, 127]|

> NVIDIA에서 발표한 논문 [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) 에서 양자화에 대한 기본적인 내용을 아주 잘 설명하고 있다.

<br>

# Per-Tensor and Per-Channel Quantization

Quantization scale은 두 가지로 나눌 수 있다.

- _Per-tensor quantization_ : in which a single scale value (scalar) is used to scale the entire tensor.
- _Per-channel quantization_ : in which a scale tensor is broadcast along the given axis - for convolutional neural networks, this is typically the channel axis.

Explicit quantization에서 weights는 per-tensor 또는 per-channel로 양자화될 수 있다. 두 경우에서 모두 precision은 FP32이다. Activations는 오직 per-tensor quantization만 가능하다.

Per-channel quantization에서 양자화 축은 output-channel axis이어야 한다. 예를 들어, 2D convlution의 weight를 `KCRS` 표기법으로 표현할 때, `K`가 output-channel axis이고 weight quantization은 다음과 같다.
```
For each k in K:singe
    For each c in C:
        For each r in R:
            For each s in S:
                output[k,c,r,s] := clamp(round(input[k,c,r,s] / scale[k]))
```
한 가지 예외는 deconvolution(known as _transposed convlution_)이며, input-channel axis가 양자화된다.

Quantization scale은 걔수(coefficients) 벡터이며, 반드시 quantization axis와 같은 크기를 갖는다. 그리고, 계수는 모두 positive float이다. 반올림 방법은 [rounding-to-nearest ties-to-even](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even)이며 [-128, 127] 범위로 clamping 된다.

Dequantization은 아래와 같이 pointwise operation이라는 점만 제외하면 유사하게 수행된다.
```
output[k,c,r,s] := input[k,c,r,s] * scale[k]
```

TensorRT는 activation 텐서에 대해서만 per-tensor quantization을 지원하지만, 아래의 경우에 대해서는 per-channel weight quantization을 지원한다.

- convolution layer
- deconvolution layer
- fully connected layer
- MatMul where the second input is constant and both input matrices are 2D

<br>

# References

- [NVIDIA TensorRT Documentation: Introducing to Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization)
- [pytorch-quantization (github)](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [Paper: Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)