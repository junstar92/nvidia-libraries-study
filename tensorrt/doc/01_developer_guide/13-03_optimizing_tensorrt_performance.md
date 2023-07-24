# Table of Contents

- [Table of Contents](#table-of-contents)
- [Optimizing TensorRT Performance](#optimizing-tensorrt-performance)
- [Batching](#batching)
- [Within-Inference Multi-Streaming](#within-inference-multi-streaming)
- [Cross-Inference Multi-Streaming](#cross-inference-multi-streaming)
- [CUDA Graphs](#cuda-graphs)
- [Enabling Fusion](#enabling-fusion)
  - [Layer Fusion](#layer-fusion)
  - [Types of Fusions](#types-of-fusions)
  - [PointWise Fusion](#pointwise-fusion)
  - [Q/DQ Fusion](#qdq-fusion)
- [Limiting Compute Resources](#limiting-compute-resources)
- [Deterministic Tactic Selection](#deterministic-tactic-selection)
- [Overhead of Shape Change and Optimization Profile Switching](#overhead-of-shape-change-and-optimization-profile-switching)
- [References](#references)

<br>

# Optimizing TensorRT Performance

이번 포스팅에서는 GPU에서의 일반적은 inference flow에 초점을 맞추고 성능을 향상시킬 수 있는 일반적인 전략들을 살펴본다.

# Batching

가장 중요한 최적화는 batching을 사용하여 가능한 한 많은 결과를 병렬로 계산하는 것이다. TensorRT에서 batch는 균일하게 처리할 수 있는 입력의 모음이다. Batch의 각 인스턴스들은 모두 같은 shape를 가지며 정확히 동일한 방식으로 네트워크를 통해 처리된다. 따라서, 각 인스턴스는 명백히 병렬로 계산될 수 있다.

네트워크의 각 레이어에는 forward inference를 계산하는데 필요한 약간의 오버헤드와 동기화가 있다. 더 많은 결과를 병렬로 계산하려면 더 효율적이다. 또한, 많은 레이어에서는 가장 작은 차원의 입력에서 성능이 제한된다. 만약 batch 크기가 1이거나 작은 경우에는 성능이 제한될 수 있다는 것을 의미한다. 예를 들어, `V` 입력과 `K` 출력의 fully connected 레이어는 `1xV` 행렬과 `VxK`의 weight 행렬의 곱으로 하나의 batch 인스턴스에 대해 구현될 수 있다. `N`개의 batch 인스턴스가 일괄적으로 처리되면 `NxV`와 `VxK` 행렬을 곱한 값이 된다. Vector-matrix multiplier보다 matrix-matrix multiplier가 더 효율적이다.

Batch가 더 크면 대부분의 경우 GPU에서 더 효율적이다. `N > 2^16`과 같이 극단적으로 큰 batch에서는 때때로 확장되는 인덱스 계산이 필요할 수 있으므로 가능하면 피해야 한다. 그러나 일반적으로 batch 크기를 증가시키면 총 throughput이 향상된다. 또한, 네트워크에 `MatrixMultiply` 레이어 또는 `FullConnected` 레이어가 포함된 경우, 하드웨어 지원한다면 Tensor Core를 활용하기 때문에 32배수의 batch 크기에서 FP16 또는 INT8 추론에 대해 최상의 성능을 갖는 경향이 있다.

NVIDIA Ada Lovelace GPUs 또는 그 이후의 GPU에서는 입력/출력 값을 L2 캐시에 캐싱할 수 있도록 도움이 되는 더 작은 batch 크기가 발생하는 경우, batch 크기를 줄이면 throughput이 크게 향상될 수 있다. 따라서 최적의 성능을 위한 batch 크기를 실험적으로 구해야 한다.

어플리케이션에 따라 batching inference가 불가능할 수 있다. Request 당 추론을 수행하는 서버와 같은 어플리케이션은 opprotunisic batching을 구현할 수 있다. 각 요청에 대해 T 시간동안 기다리고, 해당 시간 안에 다른 request가 들어오면 이들을 함께 일괄적으로 처리한다. 그렇지 않다면 단일 인스턴스로 추론을 수행하게 된다. 이러한 전략은 각 request에 고정적인 대기 시간을 갖도록 하지만 시스템의 처리량을 크게 향상시킬 수 있다.

### Using batching

네트워크가 explicit batch mode를 사용한다면 batch 차원은 텐서 차원의 일부가 되며 optimization profiles를 추가하여 batch 크기의 범위와 엔진이 최적화할 batch 크기를 지정할 수 있다. 이에 대한 자세한 내용은 [Working with Dynamic Shapes](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md)에서 다루고 있다.

네트워크가 implicit batch mode를 사용하는 경우, `IExecutionContext::execute`와 `IExecutionContext::enqueue` 메소드는 batch 크기에 대한 파라미터를 사용한다. `IBuilder::setMaxBatchSize`를 사용하여 네트워크를 빌드할 때 최대 batch 크기도 설정해야 한다. 또한, `execute` 또는 `enqueue`를 호출할 때 매개변수로 전달된 바인딩은 인스턴스가 아닌 텐서별로 구성된다. 즉, 하나의 입력 인스턴스에 대한 데이터는 하나의 연속적인 메모리 영역에 함께 그룹화되지 않는다. 대신, 각 텐서 바인딩은 해당 텐서에 대한 인스턴스 데이터의 배열이다.

또 다른 고려 사항은 최적화된 네트워크 빌드는 주어진 최대 batch 크기에 대해 최적화한다는 것이다. 최종 결과는 최대 batch 크기에 대해 튜닝되지만 그래도 더 작은 batch 크기에서도 올바르게 동작한다. 다양한 배치 크기에 대해 최적화된 엔진을 여러 개 빌드하고, 런타임에서 실제 배치 크기에 따라서 사용할 엔진을 선택할 수 있다.

# Within-Inference Multi-Streaming

일반적으로 CUDA 프로그래밍 스트림은 비동기 작업을 구성하는 방법이다. 스트림에 입력된 비동기 명령은 순서대로 실행되지만 다른 스트림에 대해서는 잘못된 순서로 실행될 수 있다. 예를 들어, 두 스트림의 비동기 명령은 동시에 실행되도록 스케줄링될 수 있다.

TensorRT의 context 및 inference에서 최적화된 네트워크의 각 레이어에는 GPU 연산이 필요하다. 그러나 모든 레이어가 하드웨어를 완전히 사용하는 것은 아니다. 별도의 스트림에서 연산을 요청하면 불필요한 동기화없이 하드웨어를 사용할 수 있도록 스케줄링할 수 있다. 일부 레이어만 오버랩할 수 있어도 전체 성능은 향상된다.

TensorRT 8.6부터 `IBuilderConfig::setMaxAuxStreams()` API를 사용하여 TensorRT가 여러 레이어를 병렬로 실행하는데 사용할 수 있는 보조 스트림(auxiliary streams)의 최대 갯수를 설정할 수 있다. 이는 `enqueueV3()` 호출에서 제공되는 "main stream"과 대조되며, 활성화된 경우 TensorRT는 main stream에서 실행되는 레이어와 병렬로 보조 스트림에서 일부 레이어를 실행한다.

예를 들어, 최대 8개의 스트림에서 추론을 수행하려면 (1 main stream and 7 auziliary streams), 아래와 같이 config를 설정한다.
```c++
config->setMaxAuxStream(7);
```

최대로 사용할 보조 스트림의 수를 설정하지만, TensorRT는 더 적은 수의 보조 스트림을 사용할 수도 있다. 아래의 코드로 TensorRT가 실제로 사용하는 보조 스트림의 수를 확인할 수 있다.
```c++
int32_t nbAuxStreams = engine->getNbAuxStreams();
```

엔진으로부터 execution context가 생성될 때, TensorRT는 자동으로 추론에 필요한 보조 스트릠을 생성한다. 그러나, 아래와 같이 직접 보조 스트림을 지정할 수도 있다.
```c++
int32_t nbAuxStreams = engine->getNbAuxStreams();
std::vector<cudaStream_t> streams(nbAuxStreams);
for (int32_t i = 0; i < nbAuxStreams; i++) {
  cudaStreamCreate(&streams[i]);
}
context->setAuxStreams(streams.data(), nbAuxStreams);
```

TensorRT는 항상 `enqueueV3()`를 호출하여 제공된 메인 스트림과 보조 스트림 사이에 이벤트 동기화를 다음과 같이 삽입한다.

- `enqueueV3()` 호출이 시작될 때, TensorRT는 모든 보조 스트림이 메인 스트림의 활동을 기다리도록 한다.
- `euqneueV3()` 호출이 끝나면, TensorRT는 메인 스트림이 보조 스트림의 활동을 대기하는지 확인한다.

보조 스트림을 활성화하면 일부 activation 버퍼를 더 이상 재활용할 수 없기 때문에 메모리 사용이 증가할 수 있다.

# Cross-Inference Multi-Streaming

Within-inference streaming 외에도 여러 execution contexts 간 streaming도 활성화할 수 있다. 예를 들어, 여러 optimization profiles로 엔진을 빌드하고 profile 당 execution context를 생성할 수 있다. 그런 다음 서로 다른 스트림에서 execution context의 `enqueueV3()`를 호출하여 병렬로 실행할 수 있도록 한다.

여러 개의 동시 스트림을 실행하면 종종 여러 스트림이 컴퓨팅 시소스를 동시에 공유하는 상황이 발생한다. 이는 네트워크가 TensorRT 엔진이 최적화되었을 때보다 추론 중에 사용 가능한 컴퓨팅 리소스가 더 적을 수 있다는 것을 의미한다. 이러한 리소스 가용성의 차이로 인해 TensorRT는 실제 런타임 조건에 대해 차선의 커널을 선택할 수 있다. 이러한 효과를 완화하기 위해 실제 런타임 조건과 유사하도록 엔진 생성 중에 사용 가능한 컴퓨팅 리소스 크기를 제한할 수 있다. 이러한 방식은 일반적으로 latency를 희생시키면서 throughput을 향상시킨다. 이에 대한 내용은 [Limiting Compute Resources](#limiting-compute-resources)에서 조금 더 자세히 다룬다.

# CUDA Graphs

[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)는 커널의 스케줄링이 CUDA에 의해 최적화되도록 하는 방식으로 커널의 시퀀스를 나타내는 방법이다. 이는 어플리케이션 성능이 커널을 큐에 넣는 데 걸리는 CPU 시간에 민감한 경우에 특히 유용할 수 있다.

TensorRT의 `euqueueV3()` 메소드는 파이프라인 중간에 CPU 상호 작용이 필요하지 않는 모델에 대해 CUDA 그래프 캡처를 지원한다. 예를 들면, 아래와 같다.
```c++
// Call enqueueV3 once after an input shape change to update internal state.
context->enqueueV3(stream);

// Capture a CUDA graph instance
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
context->enqueueV3(stream);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, 0);

// To run inference, launch the graph instead of calling enqueueV3().
for (int i = 0; i < iterations; ++i) {
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
}
```

Loops나 conditions를 포함하는 모델은 그래프가 지원되지 않는다. 이 경우에 `cudaStreamEndCapture()`는 그래프 캡처를 실패했다고 `cudaErrorStreamCapture` 에러를 반환하지만, context는 CUDA 그래프없이 정상적인 추론에 계속 사용할 수 있다.

그래프를 캡처할 때, dynamic shapes가 있을 때 2단계의 실행 전략을 고려하는 것이 중요하다.

1. Update internal state of the model to account for any changes in input size.
2. Stream work to the GPU

빌드 시 입력 크기가 고정된 모델의 경우, 첫 번쨰 단계는 필요하지 않다. 그렇지 않다면, 만약 마지막 호출 이후에 입력 크기가 변경되었다면 파생되는 속성들을 업데이트하기 위해 일부 작업들이 필요할 수 있다.

첫 번째의 작업은 캡처되도록 디자인되지 않았으며, 캡처가 성공하더라도 모델의 실행 시간이 증가할 수 있다. 따라서, 입력의 shape나 shape 텐서의 값을 변경한 후, `enqueueV3()`를 한 번 호출하여 그래프를 캡처하기 전에 지연된 업데이트를 flush한다.

TensorRT로 캡처된 그래프는 캡처된 입력의 크기와 execution context의 상태에 따라 다르다. 그래프가 캡처된 context를 수정하면 그래프를 실행할 때 undefined behavior이 발생한다. 특히 어플리케이션이 `createExecutionContextWithoutDeviceMemory()`를 사용하여 activations를 위한 메모리를 직접 제공하는 경우, 메모리의 주소도 그래프의 일부로 캡처된다. 바딩인의 위치 또한 그래프의 일부로 캡처된다.

`trtexec`를 사용하면 빌드된 TensorRT 엔진이 CUDA 그래프 캡처와 호환되는지 확인할 수 있다.

# Enabling Fusion

## Layer Fusion

TensorRT는 빌드 단걔에서 다양한 타입의 최적화를 시도한다. 첫 번째 단계에서 레이어들은 가능한 한 퓨전된다. 많은 레이어 구현은 네트워크를 생성할 때 직접 액세스할 수 없는 추가적인 파라미터와 옵션이 있다. 대신 fusion optimization 단계에서는 지원되는 연산 패턴을 감지하고 여러 레이어를 하나의 레이어로 퓨전한다.

예를 들어, Convolution 연산 다음에 ReLU activation 연산이 이어지는 일반적인 경우가 있다. 이러한 네트워크를 생성하려면 `addConvolution`으로 convolution 레이어를 추가하고, `ActivationType`이 `kRELU`인 `addActivation`을 호출하여 activation 레이어를 추가해야 한다. 최적화되지 않은 그래프에서는 convolution과 activation 레이어가 별도로 존재한다. Convolution의 내부 구현은 activation을 위한 커널 호출없이 convolution 커널에서 직접 ReLU 함수 계산을 지원한다. Fusion optimization 단계는 ReLU 연산으로 이어지는 convolution을 감지하고, 이러한 연산이 레이어 구현에서 지원되는지 확인한 다음 이를 하나의 레이어로 퓨전한다.

퓨전이 발생했는지 조사하기 위해 builder는 제공된 logger를 통해 작업들을 기록한다. 이 정보는 `kINFO` 레벨에서 출력된다.

퓨전은 일반적으로 퓨전된 두 레이어의 이름을 포함하는 이름으로 새로운 레이어를 생성하여 처리한다. 예를 들어, `ip1`이라는 FullyConnected 레이어와 `relu1`이라는 ReLU Activation 레이어가 퓨전되어 `ip1 + relu1`이라는 새로운 레이어를 생성한다.

## Types of Fusions

### Supported Layer Fusions

아래 리스트는 지원되는 퓨전 타입을 나열한다. 특별한 언급이 없다면 타입이나 값에 제한이 없다.

|Layers|Description|
|------|-----------|
|`ReLU Activation` | ReLU activaion layer 이후에 이어지는 ReLU activation은 singla activation layer로 대체됨 |
|`Convolution and ReLU Activation` | 모든 convolution type에 대해서 퓨전 적용 |
|`Convolution and GELU Activation` | input과 output의 precision은 FP16 또는 INT8로 동일해야 한다. CUDA 10.0 이상이 설치된 Turing 이상의 device에서 TensorRT가 실행되어야 한다. |
|`Convolution and Clip Activation` | 모든 convolution type에 대해서 퓨전 적용 |
|`Scale and Activation` | Scale layer 이후에 이어지는 activation layer는 single activation으로 퓨전될 수 있다. |
|`Convolution and Elementwise Operation` | Convolution layer 이후에 이어지는 sum/min/max를 수행하는 elementwise layer는 convolution layer에 퓨전될 수 있다. |
|`Padding and Convolution/Deconvolution` | Padding size가 음수가 아닌 경우, padding에 이어서 나타나는 convolution/deconvolution layer는 단일 convolution/deconvolution layer로 퓨전될 수 있다. |
|`Shuffle and Reduce` | Reshape가 없는 shuffle layer 이후에 이어지는 reduce layer는 단일 reduce layer로 퓨전될 수 있다. Shuffle layer는 permutation을 수행할 수는 있지만 reshape 연산은 수행할 수 없으며, reduce layer에는 반드시 차원의 `keepDimensions` set을 가져야 한다. |
|`Shuffle and Shuffle` | 각 shuffle layer는 transpose, reshape, second transpose로 구성되며, shuffle layer에 이어서 나오는 또 다른 shuffle layer는 단일 shuffle로 대체되거나 아무 작업도 하지 않을 수 있다. 두 shuffle layer가 모두 reshape 연산을 수행하는 경우, 첫 번째 shuffle의 second transpose가 두 번째 shuffle의 first transpose의 inverse인 경우에만 퓨전이 허용된다. |
|`Scale` | 0을 더하거나, 1을 곱하거나, 1의 거듭제곱을 계산하는 scale layer는 제거될 수 있다. |
|`Convolution and Scale` | Convolution layer 뒤에 이어지는 `kUNIFORM` 또는 `kCHANNEL`인 scale layer는 convolution layer의 weights를 조정하여 단일 convolution layer로 퓨전될 수 있다. Scale이 constant power parameter가 아니라면 이 퓨전은 비활성화된다.|
|`Convolution and Generic Activation` | 이 퓨전은 아래에서 언급되는 퓨전이 적용된 이후에 발생한다. 하나의 input과 하나의 output을 갖는 pointwise는 `generic activation layer`라고 부를 수 있다. Convolution layer 이후에 이어지는 generic activation layer는 단일 convolution layer로 퓨전될 수 있다. |
|`Reduce` | Average pooling을 수행하는 reduce layer는 pooling layer로 대체된다. Reduce layer는 `keepDimensions` set을 반드시 가져야 하며, `kAVG` 연산을 사용하여 배치 처리하기 전에 CHW input format에서 H와 W 차원에 걸쳐 reduction을 수행한다. |
|`Convolution and Pooling` | Convolution layer와 pooling layer의 precision은 동일해야 한다. Convolution layer에는 이전 퓨전에 의해서 이미 퓨전된 activation 연산이 있을 수 있다. |
|`Depthwise Separable Convolution` | Activation을 가진 depthwise convolution에 이어서 나타나는 activation을 가진 convolution은 때때로 single optimized DepSeqConvolution layer로 퓨전될 수 있다. 두 convolution의 precision은 INT8 이어야 하며, device의 compute capability는 7.2 이상이어야 한다. |
|`SoftMax and Log` | SoftMax가 이전의 log operation과 퓨전되지 않은 경우, 단일 SoftMax layer로 퓨전될 수 있다. |
|`SoftMax and TopK` | 두 레이어는 단일 레이어로 퓨전될 수 있다. 이때 SoftMax는 log operation을 포함하거나 포함하지 않을 수 있다. |
|`FullyConnected` | FullyConnected layer는 convolution layer로 변환되며, convolution의 모든 퓨전이 적용된다. |

### Supported Reduction Operation Fusions

아래 리스트는 지원되는 reduction operation fusion을 나열한다.
|Operation|Description|
|---------|-----------|
|`GELU`|아래 방정식을 나타내는 unary layer와 elementwise layer는 단일 GELU reduction operation으로 퓨전될 수 있다. <br> $0.5x \times (1 + \tanh{(2 / \pi (x + 0.044615x^3))}$ <br>or <br> $0.5x \times (1 + \text{erf}(x / \sqrt{2})$ |
|`L1Norm`| `kABS` 연산의 unary layer 이후에 이어지는 `kSUM` 연산 reduce layer는 단일 L1Norm reduction operation으로 퓨전될 수 있다. |
|`Sum of Squares`| 동일한 입력의 product elementwise layer (square operation)에 이어지는 `kSUM` reduction layer는 단일 square Sum reduction operation으로 퓨전될 수 있다. |
|`L2Norm`| Squares 연산의 합에 이어서 나타나는 `kSQRT` UnaryOperation은 단일 L2Norm reduction operation으로 퓨전될 수 있다. |
|`LogSum`| `kSUM` 연산의 reduce layer 이후에 이어지는 `kLOG` UnaryOperation은 단일 LogSum reduction operation으로 퓨전될 수 있다. |
|`LogSumExp`| `kEXP`의 unary elementwise operation에 이어서 나타나는 `LogSum`은 단일 LogSumExp reduction으로 퓨전될 수 있다. |

## PointWise Fusion

여러 인접한 PointWise 레이어는 성능 향상을 위해서 하나의 PointWise 레이어로 퓨전될 수 있다.

아래의 PointWise 레이어 타입이 지원되지만, 몇몇 제한 사항이 존재한다.

- `Activation` : 모든 `ActivationType`이 지원된다.
- `Constant` : 단일 값의 상수인 경우에만 지원된다 (size == 1).
- `ElementWise` : 모든 `ElementWiseOperation`이 지원된다.
- `PointWise` : `PointWise` 자체도 PointWise 레이어이다.
- `Scale` : `ScaleMode::kUNIFORM`에 대해서만 지원된다.
- `Unary` : 모든 `UnaryOperation`이 지원된다.

퓨전된 PointWise 레이어의 크기는 무제한이 아니다. 따라서 일부 PointWise 레이어는 퓨전되지 않을 수 있다.

퓨전이 되면 두 레이어로 구성된 이름으로 새로운 레이어를 생성한다. 예를 들어, 이름이 `add1`인 ElementWise 레이어와 이름이 `relu1`인 이름의 ReLU Activation 레이어는 `fusedPointwiseNode(add1, relu1)`이라는 이름으로 퓨전된다.

## Q/DQ Fusion

[NVIDIA Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)과 같은 QAT tool로부터 생성된 quantized INT8 그래프는 scales와 zero-points를 가진 `onnx::QuantizeLinear`와 `onnx::DequantizeLinear`의 노드 쌍(Q/DQ)으로 구성된다. TensorRT 7.0부터 `zero_point`는 `0` 이어야 한다.

Q/DQ 노드는 FP32값에서 INT8로 또는 그 반대로 변환하는데 도움을 준다. 이러한 그래프에서는 여전히 FP32 정밀도의 weights와 bias를 갖는다.

필요한 경우에 weights를 양자화/역양자화할 수 있도록 weights 다음에 Q/DQ 노드 쌍이 따라온다. Bias quantization은 activations와 weights로부터 scales를 사용하여 수행되므로, bias input에 추가적인 Q/DQ 노드 쌍은 필요하지 않다. Bias quantization에 대한 가정은 `S_weight * S_input = S_bias`이다.

Q/DQ 노드와 연관된 퓨전에는

- quantizing/dequantizing weights
- commutating Q/DQ nodes without changing the mathematical equivalence of the mode
- erasing redundant Q/DQ nodes

가 포함된다. Q/DQ 퓨전을 적용한 후, 나머지 builder optimization이 그래프에 적용된다.

### Fuse Q/DQ with weighted node (Conv, FC, Deconv)

만약 `([DQ, DQ] > Node > Q)` 시퀀스의 그래프가 있다면,
```
[DequantizeLinear (Activations), DequantizeLinear (weights)] > Node > QuantizeLinear
```
이는 quantized node `(QNode)`로 퓨전된다.

Weights에 대한 Q/DQ 노드 쌍을 지원하려면 weighted nodes가 둘 이상의 입력을 지원해야 한다. 따라서, second input (for weights tensor)와 third input (for bias tensor) 추가를 지원한다. 추가적인 입력은 Convolution, Deconvolution, FullyConnected 레이어에 대한 `setInput(index, tensor)` API를 사용하여 설정할 수 있으며, index 2는 weights 텐서이고 index 3은 bias 텐서이다.

Weights nodes와 퓨전하는 동안 FP32 weights를 INT8로 양자화하고 해당되는 weighted node와 이를 퓨전한다. 마찬가지로 FP32 bias는 INT32(?)로 양자화되고 퓨전된다.

> Bias는 FP32에서 INT32로 양자화된다고 언급하고 있다. INT8을 잘못 표기한 게 아닌가 의심되는 부분이다.

### Fuse Q/DQ with non-weighted node

`DequantizeLinear > Node > QuantizeLinear (DQ > Node > Q)` 시퀀스는 quantized node `(QNode)`로 퓨전된다.

### Commutate Q/DQ node

`DequantizeLinear` commutation은 $\phi(DQ(x)) == DQ(\phi(x))$ 일 때 허용된다. `QuantizeLinear` commutation은 $Q(\phi(x)) == \phi(Q(x))$ 일 때 허용된다.

또한, commutation logic은 mathematical equivalence를 보장하는 사용 가능한 커널 구현을 고려한다.

### Insert missing Q/DQ nodes

만약 노드에 누락된 Q/DQ 노드 쌍이 있고 max(abs($\phi(x)$)) == mas(abs($x$))라면, INT8 precision으로 더 많은 노드를 실행하기 위해 누락된 Q/DQ 쌍이 삽입된다.

### Erase redundant Q/DQ nodes

모든 최적화를 적용한 후에 그래프에서 자체적으로 아무것도 하지 않는 Q/DQ 노드가 있을 수 있다. Q/DQ node erasure fusion은 이러한 중복되는 노드 쌍을 제거한다.

# Limiting Compute Resources

엔진 생성 중에 TensorRT에서 사용할 수 있는 컴퓨팅 리소스를 제한하면 런타임에서 예상되는 조건을 더 잘 나타낼 수 있어서 유용하다. 예를 들어, GPU가 TensorRT 엔진과 병렬로 추가적인 작업을 수행할 것으로 예쌍되거나 엔진이 리소스가 적은 다른 GPU에서 실행될 것으로 예상되는 경우가 이에 해당한다.

아래의 단계를 통해 사용 가능한 컴퓨팅 리소스의 수를 제한할 수 있다.

1. Start the CUDA MPS control daemon.
   ```
   nvidia-cuda-mps-control -d
   ```
2. Set the number of compute resources to use with the `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable. For example, `export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`.
3. Build the network engine.
4. Stop the CUDA MPS control daemon.
   ```
   echo quit | nvidia-cuda-mps-control
   ```

위 과정을 통해 생성되는 엔진은 감소된 compute core (여기서는 50%)에 최적화되어 예상되는 조건에서 더 좋은 throughput을 제공한다. 다양한 수의 스트림과 다양한 MPS 값으로 실험하여 네트워크에 대한 최상의 성능을 결정하는 것이 좋다.

`nvidia-cuda-mps-control`에 대한 자세한 내용은 [nvidia-cuda-mps-control 문서](https://docs.nvidia.com/deploy/mps/index.html#topic_5_1_1)에서 다루고 있다.

# Deterministic Tactic Selection

엔진 빌드 단계에서 TensorRT는 모든 가능한 tactics를 실행하고 가장 빠른 것을 선택한다. Tactics의 latency 측정에 기반하여 선택되므로 일부 tactics의 latency가 유사한 경우, TensorRT는 빌드할 때마다 다른 tactic을 선택할 수 있다. 따라서, 동일한 `INetworkDefinition`에서 빌드된 서로 다른 엔진은 출력 값과 성능 측면에서 약간 다르게 동작할 수 있다. Engine Inspector를 사용하거나 엔진을 빌드하는 동안 상세 정보를 로깅하도록 하면 엔진에서 선택된 tactics를 확인할 수 있다.

만약 결정론적(deterministic)인 tactic selection이 필요하다면, 다음의 몇 가지 제안들이 도움이 될 수 있다.

### Locking GPU Clock Frequency

기본적으로 GPU의 clock frequency는 고정되어 있지 않다. 즉, 보통 GPU는 idle clock frequency에 있으며, 활성된 GPU workload가 있을 때만 max clock frequency로 부스트된다. 그러나 idle frequency에서 boost되기 위한 latency가 있으며 TensorRT가 tactics를 실행하고 최상의 것을 선택하는 동안 성능 변동이 발생할 수 있으므로 non-deterministic tactic selection이 발생할 수 있다.

그러므로 TensorRT 엔진을 빌드하기 전에 GPU clock frequency를 고정시키면 tactic selection의 결정성을 향상시킬 수 있다. GPU clock frequency 고정은 `sudo nvidia-smi -lpc <freq>` 커맨드 호출로 할 수 있으며, 여기서 `<freq>`는 원하는 frequency 값이다. `nvidia-smi -q -d SUPPORTED_CLOCKS` 호출로 GPU에서 지원되는 frequencies를 찾을 수 있다.

### Increasing Average Timing Iterations

기본적으로 TensorRT는 tactic을 최소 4번 실행하고 평균적인 latency를 사용한다. `setAvgTimingIteration()` API를 호출하여 반복 횟수를 늘릴 수 있다.
```c++
config->setAvgTimingIteration(8);
```

Tactic 반복 실행 횟수를 증가시키면 tactic selection의 결정성을 향상시킬 수 있지만, 엔진 빌드 시간이 길어질 수 있다.

### Using Timing Cache

Timing Cache는 특정 레이어 구성에 대한 각 tactic의 latency를 기록한다. TensorRT가 만약 동일한 구성을 가진 다른 레이어를 만나면 tactic latencies가 재사용된다. 따라서, 동일한 `INetworkDefinition` 및 builder config로 실행되는 여러 엔진 빌드에서 동일한 timing cache를 재사용하여 생성되는 엔진들이 동일한 tactics를 선택하도록 할 수 있다.

Timing Cache에 대한 내용은 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#timing-cache)에서 확인할 수 있다.

# Overhead of Shape Change and Optimization Profile Switching

`IExecutionContext`가 새로운 optimization profile로 스위칭하거나 input binding의 shape가 변경한 후, TensorRT는 네트워크 전체에서 텐서의 shape를 다시 계산하고 다음 inference가 시작되기 전에 새로운 shape에 대한 일부 tactic에 필요한 리소스를 다시 계산해야 한다. 즉, shape 또는 profile이 변경된 이후에 발생하는 첫 번째 `enqueue()` 호출은 그 다음에 이어서 나타나는 `enqueue()` 호출보다 시간이 더 걸릴 수 있다.

Shape/profile 스위칭의 비용을 최적화하는 것은 개발의 영역이다. 그러나 이 오버헤드가 inference application의 성능에 영향을 미칠 수 있는 몇 가지 경우가 여전히 존재한다. 예를 들어, NVIDIA Volta GPU 또는 이전 GPU에 대한 일부 convolution tactics는 사용 가능한 모든 tactics 중에서 가장 우수하더라도 shape/profile 스위칭 오버헤드가 훨씬 더 길다. 이런 경우에는 엔진을 빌드할 때 tactic sources로부터 `kEDGE_MASK_CONVOLUTIONS` tactics를 비활성화하면 shape/profile switching 스위칭의 오버헤드를 줄이는 데 도움이 될 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Optimizing TensorRT Performance](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-performance)