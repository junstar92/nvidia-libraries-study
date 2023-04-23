# Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [cuDNN Handle](#cudnn-handle)
  - [Tensors and Layouts](#tensors-and-layouts)
  - [Tensor Core Operations](#tensor-core-operations)
- [References](#references)

<br>

# Overview

NVIDIA CUDA Deep Neural Network Library (cuDNN)은 deep neural network(DNN)를 위한 프리미티브들의 GPU-accelerated library이다. DNN 어플리케이션에서 아래와 같이 빈번하게 사용되는 연산들에 대해 고도로 최적화된 구현을 제공한다.

- Convolution forward and backward, including cross-correlation
- Matrix multiplication
- Pooling forward and backward
- Softmax forward and backward
- Neuron activations forward and backward: `relu`, `tanh`, `sigmoid`, `elu`, `gelu`, `softplus`, `swish`
- Arithmetic, mathematical, relational, and logical pointwise operations (including various flavors of forward and backward neuron activations)
- Tensor transformation functions
- LRN, LCN, batch normalization, instance normalization, and layer normalization forward and backward

각 개별 연산의 효율적인 구현뿐만 아니라, 추가적인 최적화를 위한 유연한 multi-operation fusion 패턴도 지원한다. cuDNN은 딥러닝 사용 사례에서 NVIDIA GPU에서의 가장 효율적인 성능 달성을 목표로 한다.

cuDNN 버전 7 또는 이전 버전에서 고정된 연산 및 퓨전 패턴 셋을 지원하도록 API가 설계되었다. 비공식적으로 이를 "legacy API"라고 부른다. cuDNN 버전 8부터는 빠르게 확장되는 여러 퓨전 패턴을 해결하기 위해, 고정된 API 호출 대신 연산 그래프(operation graph)를 정의하여 계산을 표현할 수 있는 [Graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)를 추가했다. Graph API는 legacy API보다 더 유연하며, cuDNN을 사용할 때 권장하는 방법이다.

cuDNN은 C API이지만, C API를 래핑하는 C++ 레이어를 오픈소스([cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend))로 제공한다. 다만, C++ 레이어는 graph API만 지원한다.

<br>

# Core Concepts

## cuDNN Handle

cuDNN 라이브러리는 host API로 노출되며, GPU를 사용하는 작업의 경우에는 필요한 데이터를 GPU device에서 직접 액세스할 수 있다고 가정한다.

cuDNN을 사용하려면 먼저 `cudnnCreate()`를 호출하여 라이브러리 컨텍스트에 대한 핸들을 초기화해야 한다. 이 핸들은 GPU 데이터에 대해 동작하는 모든 라이브러리 함수에 명시적으로 전달된다. 어플리케이션에서 더 이상 cuDNN을 사용하지 않으면 `cudnnDestroy()`를 통해 라이브러리 핸들과 연결된 모든 리소스를 해제할 수 있다. 이러한 방식으로 사용자는 여러 host thread, GPU 및 CUDA 스트림을 사용할 때 라이브러리의 기능을 명시적으로 제어할 수 있다.

예를 들어, cuDNN 핸들을 생성하기 전에 `cudaSetDevice()`를 사용하여 서로 다른 GPU device들을 서로 다른 host thread와 연결하고, 각 host thread에서 고유한 cuDNN 핸들을 생성하여 특정 연결된 device에 대해 라이브러리 함수를 호출할 수 있다. 라이브러리를 호출할 때 다른 핸들을 전달하면, 자동으로 다른 GPU device에서 실행된다.

특정 cuDNN 컨텍스트와 연관된 device는 `cudnnCreate()` 및 `cudnnDestroy()` 호출 간에 변경되지 않는 상태로 유지한다고 가정한다. cuDNN 라이브러리가 동일한 host thread 내에서 서로 다른 GPU deivce를 사용하려면, 어플리케이션 내에서 `cudaSetDevice()`를 호출하여 사용할 device를 설정한 다음 `cudnnCreate()`를 통해 새로운 device와 연관되는 cuDNN 컨텍스트를 새로 생성해야 한다.

## Tensors and Layouts

Graph API와 legacy API를 사용할 때, cuDNN 연산은 텐서를 입력으로 받고, 텐서를 출력 결과로 생성한다.

### Tensor Descriptor

cuDNN 라이브러리에서는 데이터를 아래의 파라미터들과 함께 generic n-D tensor descriptor로 나타낸다.

- a number of dimensions from 3 to 8
- a data type (32-bit floating-point, 64 bit-floating point, 16-bit floating-point...)
- an integer array defining the size of each dimensions
- an integer array defining the stride of each dimensions

#### WXYZ Tensor Descriptor

텐서의 descriptor 포맷은 약어를 사용하여 식별되며, 각 문자는 해당 차원을 참조한다. 이 문서에 이 용어는 모든 stride는 양수이며, 문자가 참조하는 차원은 각 stride의 내림차순으로 정렬된다는 것을 암시한다.

#### 3-D Tensor Descriptor

3-D 텐서는 일반적으로 행렬 곱셈에서 많이 사용되며, B, M, N 문자를 사용한다. 여기서 B는 batch size를 의미하며, M은 행렬의 row, N은 행렬의 column의 수를 의미한다.

> 더 자세한 내용은 [CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)를 참조 바람

#### 4-D Tensor Descriptor

4-D 텐서는 2D 이미지 baches를 위한 포맷을 정의할 때 사용되며, `N, C, H, W`라는 4개의 문자를 사용하여 각 차원은 나타낸다. 각 문자는 순서대로 batch size, feature map의 수, height, width를 나타낸다. 일반적으로 아래의 포맷들로 사용된다.

- `NCHW`
- `NHWC`
- `CHWN`

#### 5-D Tensor Descriptor

5-D 텐서는 3D 이미지 batches를 위한 포맷을 정의하는데 사용되며, `N, C, D, H, W`라는 5개의 문자를 사용하여 각 차원을 나타낸다. 4-D 텐서에서 `D`가 추가되었으며, depth를 나타낸다. 일반적으로 아래의 포맷으로 사용된다.

- `NCDHW`
- `NDHWC`
- `CDHWN`

#### Others

이외에 아래의 텐서 포맷들이 더 있다.

-  `Fullly-Packed Tensors`
-  `Partially-Packed Tensors`
-  `Spartially Packed Tensors`
-  `Overlapping Tensors`

자세히 이해하지는 못했는데, 아직 사용해본 적이 없어서 기회가 될 때 cuDNN에서 텐서를 나타내는 방법들에 대해서 알아볼 예정이다.

### Data Layout Formats

이번에는 데이터 레이아웃 형식에 따라서 cuDNN 텐서가 메모리에 어떻게 배열되는지를 살펴본다.

텐서의 레이아웃 포맷을 지정하는데 권장하는 방식은 각각의 stride를 설정하는 것이다. cuDNN v7 API와 호환성을 위해 [`cudnnTensorFormat_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t) 열거형을 통해 레이아웃 포맷을 구성할 수 있는데, legacy API를 위해 제공될 뿐 v8에서는 사용되지 않는다.

#### Example Tensor

아래의 차원으로 구성되는 iamge batches에 대해서 살펴보자. 간단하게 살펴보기 위해 이미지의 픽셀값은 0부터 순서대로 정수가 채워져있다고 가정한다.

- `N` : batch size 1
- `C` : the number of channels 64
- `H` : image height 5
- `W` : image width 4

각 차원의 값이 위와 같다면, 메모리 레이아웃은 아래와 같이 구성된다.

<img src="https://docs.nvidia.com/deeplearning/cudnn/developer-guide/graphics/fig-example-x32.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

위에서 예시로 사용한 `NCHW` 차원 (1,64,5,4)를 이용하여 다른 레이아웃 포맷에서는 어떻게 되는지 바로 아래서부터 설명한다.

#### Convolution Layouts

cuDNN에서는 convolution을 위해 `NCHW`, `NHWC`, `NC/32HW/32`의 레이아웃 포맷을 지원한다.

##### NCHW Memory Layout

`NCHW` 포맷은 다음과 같이 메모리에 배열된다.

1. 첫 번째 채널(c=0)부터 시작하여 각 요소가 연속된 row-major order로 정렬된다
2. 모든 채널들의 요소가 배열될 때까지 이어지는 채널에 대해서도 정렬한다
3. 모든 채널의 요소를 배치했다면, 다음 batch에 대해서 계속 정렬한다 (if `N` > 1)

`NCHW` 메모리 레이아웃을 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJtw25%2FbtsbBPUQfQh%2Ft5DpkeljABLMrK1RLbMLO0%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

##### NHWC Memory Layout

`NHWC` 포맷은 모든 `C` 채널에 대응하는 요소를 먼저 배열하는데, 아래의 순서로 배열하게 된다.

1. 채널 0의 첫 번째 요소, 채널 1의 첫 번째 요소, ..., 마지막 채널의 첫 번째 요소를 배열한다
2. 다음으로 채널 0의 두 번째 요소, 채널 1의 두 번째 요소, ..., 마지막 채널의 두 번째 요소를 배열한다
3. 위의 방식으로 모든 요소를 배열한다
4. 만약 다음 batch가 있다면, 위의 과정을 반복한다

`NHWC` 메모리 레이아웃을 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbL4POc%2FbtsbDEyt3px%2FzjUoUWkrLJme8jmQFz8ik1%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

##### NC/32HW32 Memory Layout

`NC/32HW32`는 `NHWC`와 유사한데, 중요한 차이점이 있다. `NC/32HW32` 메모리 레이아웃에서 64개의 채널은 32개 채널의 두 그룹으로 그룹화된다. 첫 번째 그룹은 c0에서 c31로 구성되고, 두 번째 그룹은 c32에서 c63으로 구성된다. 그런 다음 각 그룹은 `NHWC` 형식으로 배치된다 (아래 그림 참조).

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJeLXM%2FbtsbBcCOPqO%2FlX151QUzqJnQjT2dzh04X0%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

`NC/xHWx` 레이아웃을 일반화하면 다음을 관찰할 수 있다.

- 오직 채널 차원 `C`만 `x` 채널들의 각 그룹으로 그룹화된다.
- `x = 1`일 때, 각 그룹은 오직 하나의 채널만 가진다. 따라서, 한 채널의 요소들은 row-major order로 연속적으로 정렬되고, 이후에 다음 그룹이 정렬된다. 즉, `NCHW` 포맷과 동일하다.
- `x = C`일 때, `NC/xHWx`는 `NHWC`와 동일하다. 즉, channel depth `C`는 하나의 그룹으로 간주된다.
- `cudnnTensorFormat_t`의 NCHW INT8x32 포맷과 NCHW INT8x4 포맷은 `N x (C/32) x H x W x 32`(32 `C`s for every `W`)와 `N x (C/4) x H x W x 4`(4 `C`s for every `W`)로 해석할 수 있다.

#### MatMul Layouts

위에서 언급했듯이, 행렬 곱셈은 3차원 텐서를 사용하며 `BMN` 차원으로 표현된다. Strides를 통해 레이아웃이 지정될 수 있으며, 다음의 두 가지 권장하는 레이아웃이 있다.

- Packed Row-major: dim `(B, M, N)` with stride `(MN, N, 1)`
- Packed Column-major: dim `(B, M, N)` with stride `(MN, 1, M)`

3-D 텐서에 대한 unpacked layout도 지원하긴 한다.

## Tensor Core Operations

cuDNN v7 라이브러리에서는 텐서 코어(Tensor Core)를 사용하여 계산 집약적 루틴의 가속화를 도입했다. 텐서 코어는 NVIDIA Volta GPU부터 지원된다.

텐서 코어 연산은 행렬 연산을 가속화한다. FP16, FP32, INT32 값으로 누적되는 텐서 코어 연산을 사용하는데, math mode를 `CUDNN_TENSOR_OP_MATH`([`cudnnMathType_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t) enumerator)으로 설정하여 라이브러리가 텐서 코어 연산을 사용한다는 것을 나타낸다. 텐서 코어를 활성화할 수 있는 이 옵션은 루틴마다 적용해야 한다.

Default math mode는 `CUDNN_DEFAULT_MATH`이며, 여기서는 텐서 코어 연산을 사용하지 않도록 한다. 텐서 코어를 사용하면 부동소수점 연산의 순서가 달라진다. 따라서, `CUDNN_DEFAULT_MATH`와 `CUDNN_TENSOR_OP_MATH`에서의 결과 수치는 다를 수 있다. 예를 들어, 텐서 코어 연산을 사용하여 두 행렬을 곱한 결과는 스칼라 부동소수점 연산의 시퀀스로 얻은 결과와 매우 비슷하지만 항상 동일하지는 않다. 이러한 이유로 cuDNN 라이브러리에서는 텐서 코어 연산을 사용하기 전에 명시적인 user opt-in이 필요하다.

그러나, 일반 딥러닝 모델을 학습하는 실험을 통해 최종 네트워크 정확도에 대해 텐서 코어 연산의 부동소수점 연산 결과 차이를 무시할 수 있음을 보여주고 있다. 결과적으로 cuDNN 라이브러리는 두 math mode를 기능적으로 구별할 수 없는 것으로 취급하여 텐서 코어 연산이 적합하지 않은 경우에는 일반적인 스칼라 부동소수점 연산으로 수행한다.

다음 연산 커널에 대해 텐서 코어 사용이 가능하다.

- Convolutions
- RNNs
- Multi-Head Attension

딥러닝 컴파일러에서 아래의 주요 가이드라인이 있다.

- 큰 패딩과 필터의 조합을 피하여 컨볼루션 연산이 텐서 코어에 적합하도록 한다.
- 입력 및 필터를 `NHWC`로 변환하고 채널 및 배치 사이즈를 8의 배수가 되도록 한다.
- 모든 텐서, 워크스페이스, 예약된 공간의 메모리가 128비트로 정렬되어 있는지 확인한다. 1024비트 정렬이 더 좋은 성능을 보여줄 수도 있다.

### Notes on Tensor Core Precision

FP16 데이터인 경우, 텐서 코어는 FP16 input에서 동작하며, FP16으로 결과를 출력하고 FP16 또는 FP32로 누적될 수 있다. FP16 multiply는 full-precision로 이루어지고, 그 결과는 `m x n x k` 차원의 행렬에 대한 내적의 다른 곱셈 결과들과 FP32 연산으로 누적된다. 아래 그림에서 이를 표현하고 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbNZBJB%2FbtsbPoXRvFP%2F9t76wj8W0YSk2xlDUzr1P1%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

FP16으로 출력하는 FP32 누적의 경우, 누적기(accumulator)의 결과는 FP16으로 down-convert 된다 (일반적으로 누적 타입의 정밀도는 출력의 타입보다 크거나 같다).

<br>

# References
- [NVIDIA cuDNN Documentation: Overview](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#overview)
- [NVIDIA cuDNN Documentation: Core Concepts](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#core-concepts)