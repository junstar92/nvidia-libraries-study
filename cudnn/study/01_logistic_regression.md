# Table of Contents

- [Table of Contents](#table-of-contents)
- [Logistic Regression](#logistic-regression)
- [Implementation using cuDNN v7 API](#implementation-using-cudnn-v7-api)
  - [Test Data 준비](#test-data-준비)
  - [Weight/Bias 준비](#weightbias-준비)
  - [cuDNN Handle](#cudnn-handle)
  - [Descriptors](#descriptors)
  - [Inference](#inference)
  - [Free Resources](#free-resources)
- [References](#references)

<br>

# Logistic Regression

이번 포스팅에서는 MNIST 데이터셋의 0과 1을 분류하는 간단한 logistic regression을 cudnn으로 구현한다. 전체적인 과정은 다음과 같다.

1. 파이토치를 활용하여 MNIST 데이터 중 0과 1을 분류하는 간단한 logistic regression 모델 학습
2. 학습된 모델의 weight를 바이너리로 저장
3. cuDNN(v7 APIs)을 사용하여 inference 구현

모델은 간단한 logistic regression 모델이며, 일반적으로 아래의 식으로 표현한다.

$$ \sigma(\textbf{x}) = \frac{1}{1 + \exp(-(\textbf{W}^\top\textbf{x} + \textbf{b}))} $$

Logistic regression과 관련된 내용들은 검색을 통해 쉽게 찾아볼 수 있고, 자세히 설명하는 페이지들이 많기 때문에 여기서 따로 다루지는 않는다. MNIST 데이터셋에 대해서도 따로 언급하지 않고 스킵한다. 이번 포스팅에서 입력 텐서의 차원과 출력 텐서의 차원은 각각 아래와 같다.

- Input Dimension: (784, )
- Output Dimension: (1, )

전체 학습 과정은 [train.ipynb](/cudnn/code/logistic_regression/train.ipynb)에서 확인할 수 있으며, 이를 모두 실행하면 학습된 모델의 weight와 테스트용 데이터를 c++에서 읽을 수 있도록 바이너리 파일로 저장한다.

<br>

# Implementation using cuDNN v7 API

cuDNN은 v8이 업데이트되면서 기존의 방식이 legacy가 되었다. 이번 포스팅에서는 기존에 어떤 방식으로 사용했는지 간단하게 살펴보기 위해서 cudnn v7 API를 사용하여 구현한다. 자세한 사용 방법을 살펴보기 위해서 cuDNN으로 logistic regression을 어떻게 구성하는지 단계별로 살펴보자.

> 전체 코드는 [logistic_regression.cpp](/cudnn/code/logistic_regression/logistic_regression.cpp)에서 확인할 수 있다.

## Test Data 준비

MNIST 데이터는 (28, 28) 차원의 grayscale 이미지이다. 파이토치 학습 코드를 보면 알겠지만, 28x28 픽셀을 벡터로 나열하기 때문에 입력 데이터의 차원은 (784,)가 된다. 그리고, 모델을 통과하여 출력된 결과는 0~1 사이의 값으로 표현되는 확률 값이다. 따라서, 출력의 차원은 (1,)이 된다. 따라서, 다음과 같이 입력과 출력의 차원을 설정한다. 이번 포스팅에서는 precision은 FP32만 고려하기 때문에 모든 값들은 `FLOAT`로 취급한다.

```c++
cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
int batch_size = 1;
int in_dim = 28 * 28;
int out_dim = 1;
printf("Input shape            : (%d, %d, %d, %d)\n", 1, in_dim, 1, 1);
```

여기서 입력 차원을 4D로 표현하여 출력하고 있다. cuDNN에서는 텐서의 차원을 4D로 표현하는 것을 권장하며, 사용하지 않는 차원의 값은 1로 설정하라고 언급하고 있다. 따라서, 여기서도 기본적으로 텐서는 4D로 구성할 예정이다.

그리고, 바이너리 파일로 저장되어 있는 데이터를 불러온다. 각 데이터는 28x28 이미지이고 precision은 `float`라는 것에 유의하여 메모리를 할당한다.
```c++
float* input1 = (float*)malloc(sizeof(float) * in_dim);
float* input2 = (float*)malloc(sizeof(float) * in_dim);
load_binary(input1, in_dim, MNIST_0_BIN_NAME.c_str());
load_binary(input2, in_dim, MNIST_1_BIN_NAME.c_str());
```
`input1`은 0에 대한 이미지이고, `input2`는 1에 대한 이미지이다 (`load_binary` 구현은 전체 코드 참조 바람).

## Weight/Bias 준비

학습된 Logistic regression 모델의 weight와 bias를 host memory로 로드한다.
```c++
float* weight = (float*)malloc(sizeof(float) * in_dim
float* bias = (float*)malloc(sizeof(float) * out_dim)
load_binary(weight, in_dim * out_dim, FC_WEIGHT_BIN_N
load_binary(bias, out_dim, FC_BIAS_BIN_NAME.c_str());
```

## cuDNN Handle

[`cudnnHandle_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnHandle_t) 타입의 핸들은 cuDNN library context를 갖고 있는 포인터이며, 연산과 관련된 거의 모든 라이브러리 함수 호출에 파라미터로 전달된다. 이 핸들은 `cudnnCreate()`로 생성할 수 있고, `cudnnDestroy()`로 제거할 수 있다. 이 컨텍스트는 오직 하나의 device와 연관되며, `cudnnCreate()`를 호출하는 시점에서의 현재 device가 사용된다. 단, 동일한 GPU device에 대해서 여러 컨텍스트를 생성할 수도 있다.

```c++
// create cudnn handle
cudnnHandle_t handle;
cudnnCreate(&handle);
```

## Descriptors

다음으로 필요한 연산과 각 연산에서 사용되는 텐서, 필터 등을 위한 descriptor를 생성하고 설정해주어야 한다. 구현할 logistic regression에서 필요한 descriptor는 다음과 같다.

- input/output descriptors
- weight/bias descriptors
- convolution descriptor
- activation descriptor (sigmoid)

여기서 convolution descriptor를 사용한 이유는 cuDNN에서 fully connected layer 연산을 지원하지 않기 때문이다. 하지만, 1x1 convolution이 사실상 fully connected layer 연산과 동일하기 때문에 cuDNN의 convolution 연산으로 fully connected 연산이 가능하다.

> 사실 fully connected layer는 input 행렬과 weight 행렬의 곱셈으로 표현할 수 있기 때문에 BLAS API인 `gemm`을 사용해도 된다.

### Input/Output Tensor Descriptors

입력 차원은 (784, )이고, 출력 차원은 (1, )이다. 하지만, cuDNN에서 텐서의 차원은 4D로 구성하는 것을 권장한다. 또한, 위에서 언급했듯이 fully connected layer 연산을 1x1 convolution 연산으로 대체할 예정이므로 입력 차원과 출력 차원을 아래와 같이 지정한다. convolution 연산을 수행한 이후에는 bias를 더하고 activation을 적용하므로 최종 출력 또한 convolution 연산의 출력의 차원과 동일하다. 4D 텐서에서 각 차원은 convolution layer 기준으로 N(batch size), C(channel), H(height), W(width)에 해당한다.

- Input Dimension: (1, 784, 1, 1)
- Output Dimension: (1, 1, 1, 1)

```c++
cudnnTensorDescriptor_t input_desc, output_desc;
int input_shape[] = { batch_size, in_dim, 1, 1 };
int output_shape[] = { batch_size, out_dim, 1, 1 };
int input_stride[] = { in_dim, 1, 1, 1};
int output_stride[] = { out_dim, 1, 1, 1};
cudnnCreateTensorDescriptor(&input_desc);
cudnnCreateTensorDescriptor(&output_desc);
cudnnSetTensorNdDescriptor(input_desc, data_type, 4, input_shape, input_stride);
cudnnSetTensorNdDescriptor(output_desc, data_type, 4, output_shape, output_stride);
```

과정은 간단하다. Tensor descriptor를 생성하고([`cudnnCreateTensorDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateTensorDescriptor)), 해당 텐서에 대한 정보를 설정([`cudnnSetTensorNdDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor))한다. 4D 텐서를 사용하므로 `cudnnSetTensorNdDescriptor()`가 아닌 [`cudnnSetTensor4dDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptor)을 사용해도 무관하며, 조금 더 직관적으로 텐서를 설정할 수 있다.

`cudnnSetTensorNdDescriptor()`에 전달되는 stride 배열은 각 차원의 원소에 액세스하기 위한 stride 크기를 의미한다. `input_stride`를 예를 들면, `input_shape`의 첫 번째 차원이 여러 개, 즉, batch 크기가 1보다 클 때, 첫 번째 batch의 첫 번째 요소에서 두 번째 batch의 첫 번째 요소까지의 stride 값은 `in_dim`이다. 이와 같이 나머지 차원들도 동일한 방식으로 stride 값이 있을 것이고, 이 값들을 지정해주면 된다.

### Weight/Bias Descriptors

다음으로 convolution layer에서 사용할 weight를 위한 descriptor와 bias를 위한 descriptor를 생성한다. 참고로 cuDNN에서 convolution 연산은 실제 convolution 연산만 해당하며, bias를 더해주는 것은 따로 연산을 수행해주어야 한다.

```c++
cudnnFilterDescriptor_t weight_desc;
cudnnTensorDescriptor_t bias_desc;
int weight_shape[] = { out_dim, in_dim, 1, 1};
int bias_shape[] = { out_dim, 1, 1, 1 };
int bias_stride[] = { out_dim, 1, 1, 1 };
cudnnCreateFilterDescriptor(&weight_desc);
cudnnCreateTensorDescriptor(&bias_desc);
cudnnSetFilterNdDescriptor(weight_desc, data_type, CUDNN_TENSOR_NCHW, 4, weight_shape);
cudnnSetTensorNdDescriptor(bias_desc, data_type, 4, bias_shape, bias_stride);
```

Weight는 convolution 연산에서 필터(커널이라고 부르기도 함)로 사용되고, bias는 단순히 텐서를 더해주는 연산에 사용된다. 그렇기 때문에 weight의 경우에는 `cudnnFilterDescriptor_t` 타입의 descriptor를 사용되며, bias는 텐서 덧셈 연산에서 사용되는 `cudnnTensorDescriptor_t` 타입의 descriptor를 사용한다.

필터로 사용되는 weight는 [`cudnnSetFilterNdDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilterNdDescriptor)를 사용하여 설정된다. 텐서를 설정해주는 API와 거의 비슷한데, 텐서의 포맷을 지정해주는 파라미터가 추가로 존재한다. 가능한 포맷으로는 [`cudnnTensorFormat_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t)의 값 중 하나를 선택할 수 있는데, 아래의 값 중 하나를 선택할 수 있다.

- `CUDNN_TENSOR_NCHW`
- `CUDNN_TENSOR_NHWC`

여기서는 `NCHW` 포맷을 사용한다.

### Convolution Descriptors

다음으로 convolution 연산에 대한 descriptor를 생성 및 설정한다.

```c++
cudnnConvolutionDescriptor_t conv_desc;
cudnnCreateConvolutionDescriptor(&conv_desc);
const int conv_ndims = 2;
int padding[conv_ndims] = {0,0};
int stride[conv_ndims] = {1,1};
int dilation[conv_ndims] = {1,1};
cudnnSetConvolutionNdDescriptor(conv_desc, conv_ndims, padding, stride, dilation, CUDNN_CROSS_CORRELATION, data_type);

// check output dimension after convolution op
int output_shape_by_conv[4] = {};
cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_desc, weight_desc, 4, output_shape_by_conv);
printf("Input shape            : (%d, %d, %d, %d)\n", 1, in_dim, 1, 1);
printf("Output shape after conv: (%d, %d, %d, %d)\n", output_shape_by_conv[0], output_shape_by_conv[1], output_shape_by_conv[2], output_shape_by_conv[3]);
```

1x1 convolution 연산이므로 padding은 없고, stride는 1로 설정해야 한다. Dilation은 필터(커널)의 receptive field를 조금 더 넓게 설정하기 위한 파라미터이며, `dilation convolution`이라는 키워드로 찾아보면 자세히 설명하고 있는 글들을 찾아볼 수 있다. 1x1 convolution에서는 해당되는 사항이 아니므로 dilation도 1로 설정한다.

Convolution 연산 타입은 `CUDNN_CROSS_CORRELATION`으로 설정한다. 일반적으로 많은 사람들이 알고 있는 convolution layer의 연산은 사실 수학적인 의미의 convolution 연산을 뜻하는 것이 아니며, 실제로는 `cross correlation`을 의미한다. 따라서, convolution 연산 descriptor를 설정할 때 `CUDNN_CROSS_CORRELATION`으로 설정한다.

이렇게 convolution 연산에 대한 descriptor를 설정하고 나면, 입력과 convolution 연산의 필터(weight)에 대한 descriptor를 함께 사용하여 [`cudnnGetConvolutionNdForwardOutputDim()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionNdForwardOutputDim)를 통해 convolution 연산 결과 텐서의 차원을 계산할 수 있다.

위 코드를 실행하면 아래의 출력을 확인할 수 있다.
```
Input shape            : (1, 784, 1, 1)
Output shape after conv: (1, 1, 1, 1)
```

### Algorithm Selection

cuDNN에서는 다양한 convolution 알고리즘을 선택할 수 있다. 알고리즘은 [`cudnnConvolutionFwdAlgo_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t) 중 하나를 선택할 수 있다.

명시적으로 알고리즘을 선택할 수도 있지만, 일부 알고리즘은 부가적인 메모리(device) 공간이 필요하기 때문에 필요한 메모리 공간을 쿼리해주는 과정이 필요하기도 하다. Legacy API에서는 주어진 입력 및 필터 descriptor와 convolution descriptor를 가지고 사용가능한 최적의 알고리즘을 쿼리하는 API를 제공한다:

- [`cudnnGetConvolutionForwardAlgorithm_v7()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardAlgorithm_v7) : 휴리스틱한 방법으로 모든 알고리즘 중에 사용 가능한 알고리즘을 가장 빠른 알고리즘 순으로 나열한 결과를 반환한다. 위에서 따로 언급하지는 않았지만, convolution descriptor에 설정할 수 있는 `mathType_t`의 `CUDNN_DEFAULT_MATH` 버전의 알고리즘과 `CUDNN_TENSOR_OP_MATH` 버전의 알고리즘을 모두 포함한다.
- [`cudnnFindConvolutionForwardAlgorithm()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnFindConvolutionForwardAlgorithm) : 휴리스틱한 방법이 아닌 주어진 조건에서 실제 연산을 수행하여 가장 빠른 알고리즘을 찾는다. 또한, convolution descriptor에 설정된 `mathType_t`와 `CUDNN_DEFAULT_MATH` 버전의 알고리즘을 테스트한다 (convolution descriptor에 설정된 `mathType_t`가 `CUDNN_DEFAULT_MATH`가 서로 다르다고 가정).

실제 코드에서는 다음과 같이 사용할 수 있다. 여기서는 위의 두 API를 모두 테스트했다.
```c++
int requested_algo_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
int returned_algo_count = -1;
cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, weight_desc, conv_desc, output_desc, requested_algo_count, &returned_algo_count, results);
printf("\nTesting cudnnGetConvolutionForwardAlgorithm_v7...\n");
for (int i = 0; i < returned_algo_count; i++) {
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory (Math Type: %d)\n", 
        cudnnGetErrorString(results[i].status), 
        results[i].algo, results[i].time, 
        (unsigned long long)results[i].memory,
        results[i].mathType);
}
printf("\n");

cudnnFindConvolutionForwardAlgorithm(
    handle,
    input_desc, weight_desc, conv_desc, output_desc,
    requested_algo_count, &returned_algo_count,
    results
);
printf("\nTesting cudnnFindConvolutionForwardAlgorithm...\n");
for(int i = 0; i < returned_algo_count; ++i){
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory (Math Type: %d)\n", 
        cudnnGetErrorString(results[i].status), 
        results[i].algo, results[i].time, 
        (unsigned long long)results[i].memory,
        results[i].mathType);
}
printf("\n");

// set algorithm and memory
auto algo = results[0].algo;
size_t workspace_size = results[0].memory;
void* d_workspace = nullptr;
if (workspace_size > 0) {
    cudaMalloc(&d_workspace, workspace_size);
}
```

위 코드를 실행하면 아래의 출력을 얻을 수 있다.
```
Testing cudnnGetConvolutionForwardAlgorithm_v7...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 3615808 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 213384 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 7: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory (Math Type: 0)


Testing cudnnFindConvolutionForwardAlgorithm...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.038912 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.038912 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.043936 time requiring 213384 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.144384 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.565248 time requiring 3615808 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory (Math Type: 0)
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 7: -1.000000 time requiring 0 memory (Math Type: 0)
```

Convolution descriptor에 대해 math type을 특별히 지정하지 않았기 때문에 `CUDNN_DEFAULT_MATH`가 사용되었을 것이고, 쿼리된 알고리즘의 math type은 `CUDNN_DEFAULT_MATH`로 설정된 것을 볼 수 있다. 특히, 각 알고리즘에 대한 perf 구조체에는 해당 알고리즘을 쿼리할 때, 실행 시간이 얼마나 걸렸는지도 확인해볼 수 있다. `cudnnGetConvolutionForwardAlgorithm_v7()`의 경우에는 휴리스틱한 방법을 통해 알고리즘을 쿼리하기 때문에 실제 실행 시간을 확인해도 -1로 표시하지만, `cudnnFindConvolutionForwardAlgorithm()`는 실제로 알고리즘을 실행하여 가장 빠른 알고리즘을 찾기 때문에 조건에서의 연산 시간이 출력되는 것을 볼 수 있다. 그 결과, 각 API에서 가장 빠르다고 선택된 알고리즘이 다르다는 것을 확인할 수 있다.

코드 마지막 부분에서는 가장 빠르다고 쿼리된 알고리즘을 선택하고, 해당 알고리즘이 추가 메모리 공간을 필요로 한다면 해당 메모리 공간만큼 할당해주고 있는 것을 볼 수 있다. 이렇게 선택한 알고리즘과 추가 메모리 공간은 실제 convolution 연산을 수행할 때, 인자로 전달된다.

### Activation Descriptors

마지막으로 sigmoid 연산을 위한 activation descriptor를 생성하고, 설정한다.

```c++
cudnnActivationDescriptor_t sigmoid_desc;
cudnnCreateActivationDescriptor(&sigmoid_desc);
cudnnSetActivationDescriptor(sigmoid_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0);
```

방법은 매우 간단하다. [`cudnnCreateActivationDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateActivationDescriptor)를 통해 activation을 위한 descriptor를 생성하고, [`cudnnSetActivationDescriptor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetActivationDescriptor)를 통해 어떤 activation 연산을 사용할 것인지 등을 설정한다. 사용 가능한 activation function은 다음과 같다.

- `CUDNN_ACTIVATION_SIGMOID`
- `CUDNN_ACTIVATION_RELU`
- `CUDNN_ACTIVATION_TANH`
- `CUDNN_ACTIVATION_CLIPPED_RELU`
- `CUDNN_ACTIVATION_ELU`
- `CUDNN_ACTIVATION_IDENTITY`
- `CUDNN_ACTIVATION_SWISH`

`CUDNN_PROPAGATE_NAN`은 `Nan` 값일 때 전달할 것인지 여부를 의미하며, 여기서는 그대로 전달하도록 설정한다. `cudnnSetActivationDescriptor()`의 마지막 파라미터는 `coef`인데, 설정되는 activation function에 따라 동작이 조금 다르다. Sigmoid인 경우에는 딱히 해당되지 않으므로 따로 언급하지 않으며, API 문서를 참조 바란다.


## Inference

이제 실제 inference를 하기 위한 준비가 모두 완료되었다. 이번에는 위에서 설정한 텐서, 연산 및 convolution 알고리즘 등을 사용하여 실제로 추론을 수행하는 방법을 살펴본다.

기본적으로 cuDNN으로 전달되는 입력 및 출력 데이터는 device memory에 있다고 가정한다. 따라서, forward 연산으로 전달되는 실제 데이터는 device memory에 존재해야 하며, device memory의 포인터를 인자로 전달한다. 그렇기 때문에 먼저 입력, 출력 및 weight/bias 데이터를 device memory에 먼저 준비한다.

```c++
// allocate device memory for input/output
void *d_input, *d_output, *d_weight, *d_bias;
cudaMalloc(&d_input, sizeof(float) * in_dim);
cudaMalloc(&d_output, sizeof(float) * out_dim);
cudaMalloc(&d_weight, sizeof(float) * out_dim * in_dim);
cudaMalloc(&d_bias, sizeof(float) * out_dim);

// memcpy from host to device
cudaMemcpy(d_input, input1, sizeof(float) * in_dim, cudaMemcpyHostToDevice);
cudaMemcpy(d_weight, weight, sizeof(float) * out_dim * in_dim, cudaMemcpyHostToDevice);
cudaMemcpy(d_bias, bias, sizeof(float) * out_dim, cudaMemcpyHostToDevice);
cudaMemset(d_output, 0, sizeof(float) * out_dim);
```

Device memory에 준비가 완료되었으면, 이제 실제 추론을 수행한다. 연산 순서는 다음과 같으며 각 연산들을 위한 forward API가 존재한다.

1. 1x1 Convolution ([`cudnnConvolutionForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward))
2. Add bias ([`cudnnAddTensor()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnAddTensor))
3. Sigmoid function ([`cudnnActivationForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationForward))


```c++
// inference
const float alpha1 = 1.f;
const float alpha2 = 0.f;

cudnnConvolutionForward(
    handle,
    static_cast<const void*>(&alpha1),
    input_desc, d_input,
    weight_desc, d_weight,
    conv_desc, algo, d_workspace, workspace_size,
    static_cast<const void*>(&alpha2),
    output_desc, d_output
);
cudnnAddTensor(
    handle,
    static_cast<const void*>(&alpha1),
    bias_desc, d_bias,
    static_cast<const void*>(&alpha1),
    output_desc, d_output
);
cudnnActivationForward(
    handle,
    sigmoid_desc,
    static_cast<const void*>(&alpha1),
    output_desc, d_output,
    static_cast<const void*>(&alpha2),
    output_desc, d_output
);
```

여기서 `alpha1`과 `alpha2`가 별도로 사용되는데, 이들은 각 연산에서 입력 또는 출력(feedback)에 대한 scaling 등의 역할을 수행하는 파라미터이다. 이러한 파라미터는 기본적으로 host memory로 간주하기 때문에 별도의 device memory로 할당해줄 필요는 없다. 각 연산 API 문서를 살펴보면 해당 연산이 정확히 어떻게 계산되는지 설명하고 있고, `alpha1`과 `alpha2`가 어떻게 사용되는지 살펴볼 수 있다.

위 코드에서는 digit 0에 대한 이미지를 입력으로 사용했으며, 결과는 `d_output`에 저장된다. 결과 검증을 위해 다음과 같이 코드를 작성해주었고,
```c++
float output;
cudaMemcpy(&output, d_output, sizeof(float) * out_dim, cudaMemcpyDeviceToHost);
show_digit(input1, 28, 28);
printf("Output: %.3f -> Digit %d\n\n", output, output >= 0.5f ? 1 : 0);
```
위 코드 출력은 다음과 같다.
```
............................
............................
............................
............................
.............***............
.............***............
............****............
...........******...........
..........*********.........
.........***********........
.........******..***........
........******....***.......
........****.......***......
........***........***......
........**.........****.....
.......***.........***......
.......***........****......
.......***......*****.......
.......***.....******.......
.......***..********........
........************........
........**********..........
.........*******............
...........***..............
............................
............................
............................
............................
Output: 0.065 -> Digit 0
```

Inference 결과 값은 0.065이며, 결과적으로 입력이 1일 확률이 6.5%라는 의미이므로 0이라고 분류할 수 있다.

<br>

방금까지의 추론은 convolution, bias addition, activation function 연산을 각각의 함수 호출로 수행했다. cuDNN에서는 convolution - bias add - activation 연산을 하나로 통합한 버전의 API도 제공한다.

- [`cudnnConvolutionBiasActivationForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward)

따라서, 이 API를 사용하여 아래와 같이 작성하면 위와 동일한 추론을 수행할 수도 있다.

```c++
cudaMemcpy(d_input, input2, sizeof(float) * in_dim, cudaMemcpyHostToDevice);

cudnnConvolutionBiasActivationForward(
    handle,
    static_cast<const void*>(&alpha1),
    input_desc,
    d_input,
    weight_desc,
    d_weight,
    conv_desc,
    algo,
    d_workspace,
    workspace_size,
    static_cast<const void*>(&alpha2),
    output_desc,
    d_output,
    bias_desc,
    d_bias,
    sigmoid_desc,
    output_desc,
    d_output
);

// validate output
cudaMemcpy(&output, d_output, sizeof(float) * out_dim, cudaMemcpyDeviceToHost);
show_digit(input2, 28, 28);
printf("Output: %.3f -> Digit %d\n", output, output >= 0.5f ? 1 : 0);
```

위 코드는 digit 1에 대한 추론이며, 추론 결과는 다음과 같다.
```
............................
............................
............................
............................
.................*..........
.................*..........
................**..........
................**..........
................*...........
...............**...........
...............**...........
...............**...........
..............**............
..............**............
..............**............
.............**.............
.............**.............
.............**.............
............***.............
............**..............
............**..............
...........***..............
...........**...............
...........**...............
............................
............................
............................
............................
Output: 0.842 -> Digit 1
```

결과가 0.842(84.2%)이므로 해당 입력의 digit은 1이라고 분류할 수 있으며 실제 결과와 일치한다는 것을 확인할 수 있다.

## Free Resources

마지막으로 사용이 완료된 리소스들을 해제해준다.

```c++
cudnnDestroyTensorDescriptor(input_desc);
cudnnDestroyTensorDescriptor(output_desc);
cudnnDestroyFilterDescriptor(weight_desc);
cudnnDestroyTensorDescriptor(bias_desc);
cudnnDestroyConvolutionDescriptor(conv_desc);
cudnnDestroyActivationDescriptor(sigmoid_desc);
cudnnDestroy(handle);

free(input1);
free(input2);
free(weight);
free(bias);
if (d_workspace != nullptr) cudaFree(d_workspace);
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_weight);
cudaFree(d_bias);
```

<br>

# References

- [NVIDIA cuDNN Documentation: API References](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)