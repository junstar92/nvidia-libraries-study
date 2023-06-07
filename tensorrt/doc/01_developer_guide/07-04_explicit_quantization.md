# Table of Contents

- [Table of Contents](#table-of-contents)
- [Explicit Quantization](#explicit-quantization)
- [Quantized Weights](#quantized-weights)
- [ONNX Support](#onnx-support)
- [TensorRT Processing of Q/DQ Networks](#tensorrt-processing-of-qdq-networks)
- [Q/DQ Layer-Placement Recommendations](#qdq-layer-placement-recommendations)
- [Q/DQ Limitations](#qdq-limitations)
- [Q/DQ Interaction with Plugins](#qdq-interaction-with-plugins)
- [QAT Networks Using TensorFlow](#qat-networks-using-tensorflow)
- [QAT Networks Using PyTorch](#qat-networks-using-pytorch)
- [References](#references)

<br>

# Explicit Quantization

TensorRT가 네트워크에서 Q/DQ 레이어의 존재를 감지하면, explicit-precision processing logic을 사용하여 엔진을 빌드한다.

Q/DQ 네트워크는 반드시 INT8-precision builder flag가 활성된 상태에서 빌드되어야 한다.
```c++
config->setFlag(BuilderFloag::kINT8);
```

Explicit-quantization에서 INT8에 대한 표현의 변경은 명시적이므로 INT8은 type constraint로 사용되어서는 안된다.

<br>

# Quantized Weights

Q/DQ 모델의 weights는 반드시 FP32 타입을 사용하도록 지정해야 한다. Weights는 해당 weights에서 동작하는 `IQuantizeLayer`의 scale을 사용하여 TensorRT에 의해서 양자화된다. 양자화된 weights는 엔진 파일(plan)에 저장된다. Prequantized weights를 사용할 수도 있는데, 반드시 FP32 타입을 사용하도록 지정해야 한다. Q 노드의 scale은 반드시 `1.0F`로 설정되어야 하고, DQ 노드는 실제 scale 값이어야 한다.

<br>

# ONNX Support

파이토치 또는 텐서플로우에서 Quantization Aware Training (QAT)를 사용하여 학습된 모델을 ONNX로 추출할 때, 프레임워크의 그래프에서 각 fake-quantization 연산은 [QuantizeLinear](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear)와 [DequantizeLinear](https://github.com/onnx/onnx/blob/master/docs/Operators.md#dequantizelinear)라는 ONNX 연산들의 쌍으로 추출된다.

TensorRT가 ONNX 모델을 파싱하여 읽을 때, ONNX의 `QuantizeLinear` 연산은 `IQuantizeLayer` 인스턴스로 임포트되고, `DequantizeLinear` dustksdms `IDequantizeLayer` 인스턴스로 임포트된다. opset 10을 사용하는 ONNX에서 QuantizeLinear/DequantizeLinear를 지원하기 시작했으며, quantization-axis 속성(required for per-channel quantization)은 opset 13에서 추가되었다. Pytorch 1.8부터 opset 13을 사용하는 ONNX로 모델을 추출할 수 있다.

<br>

ONNX의 GEMM 연산자은 채널 별로 양자화되는 한 가지 예이다. 파이토치의 `torch.nn.Linear` 레이어는 `(K, C)` weights layout과 `transB` 속성이 활성화된 ONNX GEMM 연산자로 추출된다 (GEMM 연산이 수행되기 전에 weight를 tranpose한다). 반면, 텐서플로우에서는 ONNX로 추출되기 전에 `(C, K)` weights를 pretranspose한다.

- PyTorch: $y = \textnormal{x}W^\top$
- TensorFlow: $y = \textnormal{x}W$

따라서, 파이토치의 weights는 TensorRT에 의해서 전치된다. 전치되기 전에 weights는 TensorRT에 의해서 양자화되고, 그래서 파이토치에서 추출된 ONNX QAT 모델로부터 생성된 GEMM 레이어는 per-channel quantization에 dimension `0`을 사용한다 (axis `K = 0`). 반면 텐서플로우로부터 얻은 모델은 dimension `1`을 사용한다 (axis `K = 1`).

TensorRT는 NT8 텐서 또는 양자화된 연산자를 사용하여 이미 양자화된 ONNX 모델을 지원하지 않는다. 특히, 아래 나열된 ONNX의 양자화 연산자는 지원되지 않으며, TensorRT가 이러한 연산자가 포함된 ONNX 모델을 임포트할 때 에러가 발생한다.

- [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearConv) / [QLinearMatmul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearMatMul)
- [ConvInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvInteger) / [MatmulInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMulInteger)

<br>

# TensorRT Processing of Q/DQ Networks

TensorRT가 Q/DQ-mode에서 네트워크를 최적화할 때, 최적화 프로세스는 네트워크의 arithmetic correctness를 변경하지 않는 최적화로 제한된다. 부동소수점 연산의 순서로 인해 다른 결과가 나올 수 있으므로 bit-level의 정확도는 거의 불가능하다 (ex, a * s + b * s를 (a+b) *  a로 rewrite하는 것은 유효한 최적화이다). 이러한 차이를 허용하는 것은 일반적으로 backend optimization에서는 기본(fundamental)이며, INT8 연산을 사용하기 위해서 Q/DQ 레이어가 있는 그래프를 변환하는 데에도 이 내용이 적용된다.

Q/DQ 레이어는 네트워크의 compute 및 data precision을 제어한다. `IQuantizeLayer` 인스턴스는 quantization을 사용하여 FP32 텐서를 INT8 텐서로 변환하고, `IDequantizeLayer` 인스턴스는 dequantization을 통해 INT8 텐서를 PF32 텐서로 변환한다. TensorRT는 quantizable-layers의 각 입력에서 Q/DQ 레이어 쌍을 기대한다. Quantizable-layers는 `IQuantizeLayer` 및 `IDequantizeLayer` 인스턴스와 결합(fusion)하여 양자화된 레이어로 변환할 수 있는 deep-learning layers이다. TensorRT가 이러한 fusion을 수행할 때, quantizable layers를 실제로 INT8 data에 대해 연산하는 quantized layers로 바꾼다.

<br>

아래에서 사용되는 다이어그램에서 녹색은 INT8 precision을 나타내고, 파란색은 floating-point precision을 나타낸다. 화살표는 network activation tensors를 나타내고 사각형 박스는 network layers를 나타낸다.

아래 그림은 quantizable `AveragePool` 레이어(in blue)가 DQ 레이어 및 Q 레이어와 fusion되는 것을 보여준다. 3개의 레이어는 quantized `AveragePool` 레이어(in green)으로 대체된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq.PNG" height=250px style="display: block; margin: 0 auto; background-color:white"/>

네트워크가 최적화되는 동안, TensorRT는 Q/DQ propagation이라는 프로세스에서 Q/DQ 레이어를 이동(move)한다. Propagation의 목표는 그래프에서 낮은 정밀도로 처리할 수 있는 비율을 최대화하는 것이다. 따라서, TensorRT는 Q 노드를 뒤로 전파하고(양자화가 가능한 한 빨리 발생하도록) DQ 노드를 앞쪽으로 전파한다(역양자화가 가능한 한 늦게 발생하도). Q 레이어는 quantization과 상호 작용하는 레이어와 위치를 바꿀 수 있고, DQ 레이어는 dequantization과 상호 작용하는 레이어와 위치를 바꿀 수 있다.

- A layer `Op` commutes with quantization if `Q(Op(x)) == Op(Q(x))` (quantization과 상호 작용하는 레이어 `Op`).
- A layer `Op` commutes with dequantization if `Op(DQ(x)) == DQ(Op(x))` (dequantization과 상호 작용하는 레이어 `Op`)

아래 그림은 DQ forward-propagation과 Q backward-propagation을 설명한다. `MaxPool`에는 INT8 구현이 존재하고 DQ 및 Q와 상호 작용하는 레이어이므로, 모델을 그림과 같이 rewrite하는 것에는 문제가 없다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-propagation.PNG" height=300px style="display: block; margin: 0 auto; background-color:white"/>

> 문서에서 `MaxPool` commutation에 대해 자세하게 설명한다. 

<br>

Quantizable-layers와 commuting-layers이 처리되는 방식에는 차이점이 있다. 두 타입의 레이어는 모두 INT8로 연산할 수 있지만, quantizable-layers도 DQ input layers 및 Q output layer와 fusion된다. 예를 들어, `AveragePooling` 레이어(quantizable)는 Q 또는 DQ와 commute하지 않기 때문에 위의 첫 번째 다이어그램에서와 같이 Q/DQ fusion을 사용하여 양자화된다. 이는 `MaxPool`(commuting)이 양자화되는 방식가 대조된다.

<br>

# Q/DQ Layer-Placement Recommendations

네트워크에서 Q/DQ 레이어의 배치는 성능과 정확도에 영향을 미친다. 공격적인 양자화는 양자화로 인한 에러에 의해서 모델의 정확도가 저하될 수 있다. 그러나 양자화를 통해 latency가 감소될 수 있다. 아래에 나열된 내용은 네트워크에서 Q/DQ 레이어 배치에 대한 몇 가지 권장 사항이다.

- **Quantize all inputs of weighted-operations** (Convolution, Transposed Convolution and GEMM).

Weights와 activations에 대한 양자화는 bandwidth 요구사항을 감소시키고, INT8 연산을 사용하여 bandwidth-limited layer와 compute-limited layer를 가속화할 수 있다.

> SM 7.5 이하의 device에서는 모든 레이어에서 INT8 구현을 제공하지 않는다. 이 경우, 엔진을 빌드할 때, `could not find any implementation` 에러가 발생할 수 있다. 이를 해결하려면 해당 레이어를 양자화하는 Q/DQ 레이어를 제거하면 된다.

아래 그림은 TensorRT가 convolutional layer를 fusion하는 두 가지 예제를 보여준다. 왼쪽은 input만 양자화되고, 오른쪽은 input과 output 모두 양자화된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement1.PNG" height=400px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- **By default, do not quantize the outputs of weighted-operations**.

때때로, 더 높은 정밀도의 dequantized output을 유지하는 것이 유용하다. 예를 들어, linear operation 이후에 정확도를 위해 higher precision input이 요구되는 activation function(아래 그림에서는 `SiLU`)이 이어지는 경우가 이에 해당한다 (아래 그림 참조).

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement2.PNG" height=400px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- **Do not simulate batch-normalization and ReLU fusions in the training framework** because TensorRT optimizations guarantee to preserve the arithmetic semantics of these operations.

Batch normalization은 pre-fusion entwork에서 정의된 것과 동일한 실행 순서를 유지하면서 convolution 및 ReLU와 fusion된다. 학습 네트워크에서의 BN-folding은 필요없다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement3.PNG" height=400px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- **Quantize the residual input in skip-connections**.

TensorRT는 ResNet과 EfficientNet과 같이 skip connections가 있는 모델에서 유용한 elemen-wise addition(weighted layer 이후에 나오는)를 fusion할 수 있다. Element-wise addition layer의 첫 번째 입력의 정밀도가 fusion output의 정밀도를 결정한다.

예를 들어, 아래 다이어그램에서 $x_f^1$ 의 정밀도는 floating-point이며, 따라서, fused convolution의 output은 floating-point로 제한되고 이후의 Q 레이어는 convolution과 fusion될 수 없다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement4.PNG" height=400px style="display: block; margin: 0 auto; background-color:white"/>

대조적으로, 아래 그림에서 $x_f^1$ 은 INT8로 양자화되고 fused convolution의 output 또한 INT8이다. 따라서, 이어지는 Q 레이어가 convolution과 fusion된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement5.PNG" height=500px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- For extra performance, **try quantizing layers that do not commute with Q/DQ**.

현재 INT8 inputs의 non-weighted layers는 INT8 output을 요구한다. 따라서 input과 output 모두 양자화한다.

아래 그림은 quantizable operation을 양자화하는 예시를 보여준다. Element-wise addition은 input DQs와 output Q와 fusion된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-placement6.PNG" height=500px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- Performance can decrease if TensorRT cannot fuse the operations with the surrounding Q/DQ layers, so **be conservative when addint Q/DQ nodes and experiment with accuracy and TensorRT performance** in mind.

아래 그림은 extra Q/DQ 연산으로 발생할 수 있는 suboptimal fusions의 한 예시이다 (즉, 최적의 fusion이 아니다). 현재 위치에서 위로 두 번째 그림과 대조할 수 있는데, 위 그림이 더 좋은 성능을 보여준다. Convolution은 각 레이어들이 Q/DQ 쌍으로 둘러싸여 있기 때문에 element-wise addition과 분리되어 fusion된다. 최적의 fusion은 아래 그림의 B이며, element-wise addition의 fusion은 바로 위의 그림에서 보여주고 있다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/sub-optimal.PNG" height=500px style="display: block; margin: 0 auto; background-color:white"/>

<br>

- **Use per-tensor quantization for activations; and per-channel quantization for weights**.

이 구성이 최상의 quantization accuracy를 이끌어내는 것으로 경험적으로 입증되었다.

FP16을 활성화하여 엔진의 latency를 더욱 최적화할 수 있다. TensorRT는 가능하다면 FP32 대신 FP16을 사용하려고 시도한다 (모든 레이어 타입에 대해서 지원되지는 않음).

<br>

# Q/DQ Limitations

TensorRT가 수행하는 몇 가지 Q/DQ graph-rewrite 최적화는 2개 이상의 Q/DQ 레이어 간의 quantization scale 값을 비교하고, 만약 비교된 quantization scale이 동일한 경우에만 graph-rewrite를 수행한다. Refittable engine이 다시 fitting될 때, Q/DQ 노드의 scale에 새로운 값이 할당될 수 있다. Q/DQ engines의 refitting 연산 중에 TensorRT는 scale-dependent한 최적화에 참여한 Q/DQ 레이어에 rewrite optimization을 깨는 새로운 값이 할당되었는지 확인하고, true라면 예외를 발생시킨다.

아래 그림은 Q1과 Q2의 스케일이 같을 때를 보여준다. 만약 스케일 값이 같다면, 이들은 backward로 전파할 수 있다. 만약 엔진에 `Q1 != Q2`가 되도록 Q1 및 Q2에 새로운 값으로 refitting되면 예외가 발생하여 refitting process가 중단된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/q-dq-limitations.PNG" height=500px style="display: block; margin: 0 auto; background-color:white"/>

<br>

# Q/DQ Interaction with Plugins

플러그인은 커스텀 및 독점적(proprietary) 구현의 레이어로 교체하도록 함으로써 TensorRT의 기능을 확장한다. 사용자는 플러그인에 포함할 기능과 TensorRT에서 처리할 기능을 결정할 수 있다.

Q/DQ 레이어를 포함하는 TensorRT에도 이는 동일하게 적용된다. 플러그인이 양자화된 INT8 입력을 받고 INT8 출력을 생성할 때, input DQ와 output Q 노드는 반드시 플러그인의 일부로 포함되어 네트워크에서 제거해야 한다.

아래와 같이 하나의 INT8 플러그인을 포함하는 sequential graph를 예시로 살펴보자. INT8 플러그인(`MyInt8Plugin`)은 두 convolution 레이어 사이에 위치하고 있다 (weights quantization은 무시한다).

`Input > Q -> DQ > Conv > Q -> DQ_i > MyInt8Plugin > Q_o -> DQ > Conv > Output`

`>` 화살표는 FP32 정밀도의 activation 텐서를 나타내고, `->` 화살표는 INT8 정밀도를 나타낸다.

TensorRT가 이 그래프를 최적화할 때, 아래와 같이 레이어를 퓨전한다 (대괄호로 나타냄).

`Input > Q -> [DQ > Conv > Q] -> DQ_i > MyInt8Plugin > Q_o -> [DQ > Conv] > Output`

위 그래프에서 플러그인은 FP32 정밀도의 입력을 받고, FP32 정밀도의 출력을 내보낸다. `MyInt8Plugin` 플러그인은 INT8 정밀도를 사용하므로, 다음 단계에서는 수동으로 `DQ_i`와 `Q_o`를 `MyInt8Plugin`과 퓨전하는 것이다. 따라서, TensorRT는 이 네트워크를 다음과 같이 보게 된다.

`Input > Q -> DQ > Conv > Q -> MyInt8Plugin -> DQ > Conv > Output`

이는 다음과 같이 퓨전된다.

`Input > Q -> [DQ > Conv > Q] -> MyInt8Plugin -> [DQ > Conv] > Output`

`DQ_i`를 수동으로 fusion할 때, input quantization scale을 취하고 이를 플러그인에 전달한다. 따라서, 필요하다면 input을 dequantization하는 방법을 플러그인이 알 수 있다. 동일한 방법이 `Q_o`에도 적용된다.

<br>

# QAT Networks Using TensorFlow

TensorFlow 2 Keras 모델에서 QAT를 수행할 수 있는 오픈소스 [TensorFlow-Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)을 제공한다. 자세한 내용은 [TensorFlow-Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/index.html)을 참조하면 된다.

<br>

# QAT Networks Using PyTorch

PyTorch 1.8부터 per channel scale을 지원하는 ONNX `QuantizeLinear`/`DequantizeLinear`를 지원한다. INT8 calibration, QAT, fine-tuning을 [pyTorch-quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)를 사용하여 수행할 수 있고, ONNX로 추출할 수 있다. 자세한 내용은 [PyTorch-Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html)에서 살펴볼 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks)
- [ONNX Operation: QuantizeLinear](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear)
- [ONNX Operation: DequantizeLinear](https://github.com/onnx/onnx/blob/master/docs/Operators.md#dequantizelinear)
- [Tensorflow-Quantization Toolkit (github)](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)
- [TensorFlow-Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/index.html)
- [PyTorch-Quantization (github)](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [PyTorch-Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html)