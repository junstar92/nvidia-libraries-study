# Table of Contents

- [Table of Contents](#table-of-contents)
- [C++](#c)
- [Python](#python)
- [References](#references)

<br>

# C++

공식 문서에서는 주로 ONNX parser를 사용하여 ONNX 포맷으로부터 네트워크를 로드하는 예제를 보여준다. 하지만 TensorRT에서는 parser를 사용하는 대신 Network Definition API를 사용하여 네트워크를 직접 정의할 수도 있다. 이 경우에는 네트워크의 weights를 host memory에 미리 준비하여 네트워크 생성 중에 전달할 수 있다고 가정한다. 아래 예제 코드에서는 Input, Convolution, Pooling, MatrixMultiply, Shuffle, Activation, 그리고 Softmax Layers로 구성된 간단한 네트워크를 생성하는 방법을 보여준다.

> 아래 예제 코드에서 weights는 `weightMap`이라는 데이터 구조체에 로드되어 있다고 가정한다.

<br>

먼저 builder와 network 객체를 생성한다. 예제 코드에서 사용되는 로거는 TensorRT 샘플 코드의 [logger.cpp](https://github.com/NVIDIA/TensorRT/blob/main/samples/common/logger.cpp) 파일에 구현되어 있는 것을 사용한다고 가정한다. TensorRT 샘플에는 여러 가지 헬퍼 클래스 및 헬퍼 함수가 [common.h](https://github.com/NVIDIA/TensorRT/blob/main/samples/common/common.h) 헤더 파일에서 제공된다.
```c++
auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
const auto explicitBatchFlag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatchFlag));
```

> [Explicit Versus Implicit Batch](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)에서 `kEXPLICIT_BATCH` 플래그에 대한 내용을 자세히 살펴볼 수 있다. 일반적으로 explicit batch가 아니라면 제한되는 부분이 많은 것 같고, 거의 대부분 explicit batch를 사용하는 것으로 보인다.

다음으로 Input 레이어를 네트워크에 추가하는데, 이때, 텐서의 이름을 지정하고 텐서의 차원도 지정한다. 네트워크는 여러 입력을 가질 수 있는데, 예제 코드에서는 하나의 입력만 네트워크에 추가한다.
```c++
auto data = network->addInput(INPUT_BLOB_NAME, datatype, Dims4{1, 1, INPUT_H, INPUT_W});
```

이번에는 Convolution 레이어를 추가한다. 파라미터에는 hidden layer input nodes, strides, filter와 bias의 weights를 전달한다.
```c++
auto conv1 = network->addConvolution(
    *data->getOutput(0), 20, DimsHW{5,5}, weightMap["conv1filter"], weightMap["conv1bias"]
);
conv1->setStride(DimsHW{1,1});
```
> 위에서 사용한 convolution layer(`conv1`) API 이외에도 다양한 속성(group conv, dilation)들을 설정할 수 있는 API를 제공한다.

> TensorRT로 전달되는 weights는 host memory에 위치한다고 가정한다.

이번에는 Pooling 레이어를 추가한다. 첫 번째로 전달되는 인자는 Pooling 레이어의 입력이며, 이전 레이어의 output이다.
```c++
auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2,2});
pool1->setStride(DimsHW{2,2});
```

행렬 곱셈 연산을 위해서 입력 텐서를 reshape하기 위한 Shuffle 레이어를 추가한다.
```c++
int32_t const batch = input->getDimensions().d[0];
int32_t const mmInputs = input->getDimensions().d[1] * input->getDimensions().d[2] * input->setDimensions().d[3];
auto inputReshape = network->addShuffle(*input);
inputReshape->setReshapeDimensions(Dims{2, {batch, mmInputs}});
```

이제, MatrixMultiply 레이어를 추가한다. 여기서 transposed weights가 제공되므로, `kTRANSPOSE` 옵션을 지정해주고 있다.
```c++
IConstantLayer* filterConst = network->addConstant(Dims{2, {nbOutputs, mmInputs}}, weightMap["ip1filter"]);
auto mm = network->addMatrixMultiply(*inputReshape->getOutput(0), MatrixOperation::kNONE, *filterConst->getOutput(0), MatrixOperation::kTRANSPOSE);
```

다음으로 bias를 추가한다. 이 연산은 batch 차원에 브로드캐스트된다.
```c++
auto biasConst = network->addConstant(Dims{2, {1, nbOutput}}, weightMap["ip1bias"]);
auto biasAdd = network->addElementWise(*mm->getOutput(0), *biasConst->getOutput(0), ElementWiseOperation::kSUM);
```

> 위의 두 연산(mm + biasAdd)는 Fully Connected 레이어 하나로 사용할 수 있다.

그 다음으로 ReLU activation과 마지막 확률 계산을 위한 SoftMax 레이어를 추가해준다.
```c++
auto relu1 = network->addActivation(*biasAdd->getOutput(0), ActivationType::kRELU);
auto prob = network->addSoftMax(*relu1->getOutput(0));
```

그리고 SoftMax 레이어 출력 텐서의 이름을 추가해준다. 그래야 inference time에 memory buffer에 바인딩할 수 있다.
```c++
prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
```

이렇게 구현한 네트워크의 텐서를 출력으로 마킹해주면 네트워크 구성이 마무리된다.
```c++
network->markOutput(*prob->getOutput(0));
```

이렇게 구성한 네트워크는 엔진으로 빌드하고, 빌드한 엔진은 추론하는데 사용할 수 있다. 빌드와 추론 방법은 [Building an Engine](/tensorrt/doc/01_developer_guide/03_the_cpp_api.md#building-an-engine)과 [Deserializing a Plan](/tensorrt/doc/01_developer_guide/03_the_cpp_api.md#deserializing-a-plan)에서 설명하고 있다.

<br>

# Python

> 파이썬에서 네트워크를 구성하는 방법은 문서([link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create_network_python))에서 살펴볼 수 있다. [Jupyter Notebook](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/network_api_pytorch_mnist)에서도 자세히 설명하고 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Creating a Network Definition from Scratch](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create-network-def-scratch)