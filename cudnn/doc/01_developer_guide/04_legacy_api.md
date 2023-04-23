# Table of Contents
- [Table of Contents](#table-of-contents)
- [Convolution Functions](#convolution-functions)
  - [Prerequisites](#prerequisites)
  - [Supported Algorithms](#supported-algorithms)
  - [Data and Filter Formats](#data-and-filter-formats)
- [RNN Functions](#rnn-functions)
- [Tensor Transformations](#tensor-transformations)
  - [Conversion Between FP32 and FP16](#conversion-between-fp32-and-fp16)
  - [Padding](#padding)
  - [Folding](#folding)
  - [Conversion Between NCHW and NHWC](#conversion-between-nchw-and-nhwc)
- [Mixed Precision Numerical Accuracy](#mixed-precision-numerical-accuracy)
- [References](#references)

> 문서에서는 언급하고 있지 않지만, 아래의 내용들은 legacy API에서 각 함수에서의 텐서 코어 연산에 대해 설명하고 있다.

<br>

# Convolution Functions

## Prerequisites

텐서 코어를 지원하는 GPU에서 convolution 함수에서 텐서 코어 연산을 트리거하려면 적절한 convolution desceriptor에 대해 [`cudnnSetConvolutionMathType()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionMathType)를 호출하여 math type을 `CUDNN_TENSOR_OP_MATH` 또는 `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`으로 지정해주어야 한다.

## Supported Algorithms

위의 [전제조건](#prerequisites)을 만족할 때, 아래 convolultion 함수들은 텐서 코어 연산으로 동작할 수 있다.

- [`cudnnConvolultionForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward)
- [`cudnnConvolutionBackwardData()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData)
- [`cudnnConvolutionBackwardFilter()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter)

각 함수에서 지원되는 알고리즘은 다음과 같다.

|Supported Convolution Function|Supported Algos
|:--|:--|
|`cudnnConvolutionForward`|`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`<br>`CUDNN_CONVOLUTION_FWD_WINOGRAD_NONFUSED`|
|`cudnnConvolutionBackwardData`|`CUDNN_CONVOLUTION_BWD_DATA_ALGO_1`<br>`CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED`|
|`cudnnConvolutionBackwardFilter`|`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1`<br>`CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED`|

## Data and Filter Formats

cuDNN 라이브러리는 텐서 코어 연산을 호출하기 위해 padding, folding, NCHW-to-NHWC transformation을 사용할 수도 있다. 이에 대한 내용은 [Tensor Transformations](#tensor-transformations)를 참조 바람.

`*_ALGO_WINOGRAD_NONFUSED` 이외의 알고리즘의 경우에는 아래 요구 사항을 만족하면 cuDNN 라이브러리가 텐서 코어 연산을 트리거한다.

- Input, filter, output descriptors (`xDesc`, `yDesc`, `wDesc`, `dxDesc`, `dyDesc`, `dwDesc`)의 데이터 타입(`dataType`)은 `CUDNN_DATA_HALF` 이어야 한다. 즉, FP16 타입이어야 한다.
- Input과 output feature maps의 수, 즉, channel dimension `C`는 8의 배수이어야 한다.
- Filter의 타입은 `CUDNN_TENSOR_NCHW` 또는 `CUDNN_TENSOR_NHWC`이어야 한다.
- Filter의 타입이 `CUDNN_TENSOR_NHWC`라면, input, filter, output의 data pointer는 128-bit boundaries로 정렬되어야 한다.

<br>

# RNN Functions

<br>

# Tensor Transformations

cuDNN 라이브러리 몇몇 함수들은 실제 함수 연산을 수행하는 동안 내부에서 folding, padding, NCHW-to-NHWC 변환 등의 transformation을 수행할 수 있다.

## Conversion Between FP32 and FP16

텐서 코어 연산을 사용하기 위해서 cuDNN API에서는 FP32의 입력 데이터를 내부적으로 복사하여 FP16 데이터로 변환하도록 지정할 수 있다. 이는 [`cudnnMathType_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t) 파라미터를 `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`으로 지정하면 된다. 이 모드에서 FP32 텐서는 내부적으로 FP16으로 변환되어 텐서 연산이 수행되고, 다시 FP32로 변환되어 출력한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpXxK2%2FbtsbRKsIkUA%2FyC5KUroO1dMWqYpf9c4Ed0%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

### For Convolutions

Convolution에서는 [`cudnnSetConvolutionMathType()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionMathType)에 `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`을 전달하여 호출하면 FP32-to-FP16 변환을 수행하도록 할 수 있다.

```c++
// Set the math type to allow cuDNN to use Tensor Cores:
cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
```

### For RNNs

RNN은 [`cudnnSetRNNMatrixMathType()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNMatrixMathType)에 `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`을 전달하여 호출하면 FP32-to-FP16 변환을 수행하도록 할 수 있다. 

```c++
// Set the math type to allow cuDNN to use Tensor Cores:
cudnnSetRNNMatrixMathType(cudnnRnnDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
```

## Padding

NCHW 데이터에 대해 채널의 차원이 8의 배수가 아니라면, cuDNN 라이브러리는 텐서에 padding을 적용하여 텐서 코어 연산이 가능하도록 한다. 이러한 padding은 `CUDNN_TENSOR_OP_MATH`와 `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` 모드에서 모두 자동으로 적용된다 (packed NCHW data에 대해서만).

## Folding

Folding 또는 channel-folding을 사용하여 내부 workspace에서 입력 텐서의 formatting을 수행하여 전체 연산을 가속화할 수 있다. 이러한 변환은 convolution stride에 재한된 커널을 사용하는데 사용될 수 있다.

## Conversion Between NCHW and NHWC

텐서 코어를 사용하려면 텐서가 NHWC data layout이어야 한다. NCHW와 NHWC 간의 변환은 사용자가 Tensor Op math를 요구할 때 수행된다. 그러나, 텐서 코어를 사용하기 위한 요청은 단지 요청일 뿐이며 항상 텐서 코어로 연산한다는 것을 의미하지 않으며, 경우에 따라 텐서 코어가 사용되지 않을 수도 있다. cuDNN 라이브러리에서는 텐서 코어가 요청되었을 때, 실제로 텐서 코어가 사용되는 경우에만 NCHW와 NHWC 간의 변환을 수행한다.

<br>

# Mixed Precision Numerical Accuracy

계산의 정밀도와 결과의 정밀도가 같지 않으면 수치적 정확도가 알고리즘마다 다를 수 있다.

예를 들어, 계산은 FP32에서 수행되고 출력은 FP16인 경우, `CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 (ALGO_0)`은 `CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 (ALGO_1)`에 비해 정확도가 낮다. 이는 `ALGO_0`은 추가적인 workspace를 사용하지 않고 중간 결과를 FP16으로 누적하기 때문에 정확도가 떨어진다. 반면, `ALGO_1`은 추가 workspace를 사용하여 FP32에 중간 결과를 누적한다.

<br>

# References
- [NVIDIA cuDNN Documentation: Legacy API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#legacy-api)