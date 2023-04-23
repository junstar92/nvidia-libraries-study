# Table of Contents
- [Table of Contents](#table-of-contents)
- [Thread Safety](#thread-safety)
- [Reproducibility (Determinism)](#reproducibility-determinism)
- [Scaling Parameters](#scaling-parameters)
- [cuDNN API Compatibility](#cudnn-api-compatibility)
- [Deprecation Policy](#deprecation-policy)
- [GPU And Driver Requirements](#gpu-and-driver-requirements)
- [Convolutions](#convolutions)
- [Environment Variables](#environment-variables)
- [References](#references)

<br>

# Thread Safety

cuDNN 라이브러는 thread-safe 하다. 스레드들이 동일한 cuDNN handle을 동시에 공유하지 않는 한, 여러 호스트에서 함수를 호출할 수 있다.

스레드 별로 cuDNN handle을 생성할 때, 각 스레드가 handle을 비동기로 생성하기 전에 `cudnnCreate()`을 동기로 호출하는 것을 권장한다.

서로 다른 스레드에서 동일한 device를 사용하는 경우, 권장되는 프로그래밍 모델은 스레드 당 하나의 cuDNN handle을 생성하고 해당 handle을 스레드의 수명동안 사용하는 것이다.

<br>

# Reproducibility (Determinism)

설계상, 동일한 아키텍처의 GPU에서 주어진 버전의 cuDNN 대부분 루틴은 동일한 bit-wise 결과를 생성하지만, 몇 가지 예외가 있다. 예를 들어, 아래의 루틴은 동일한 아키텍처에서도 reproducibility를 보장하지 않는다. 이는 truly random floating point rounding errors를 도입하는 방식으로 atomic 연산을 사용하기 때문이다.

- `cudnnConvolutionBackwardFilter`에 `CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0` 또는 `CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3`를 사용할 때
- `cudnnConvolutionBackwardData`에 `CUDNN_CONVOLUTION_BWD_DATA_ALGO_0`을 사용할 때
- `cudnnPoolingBackward`에 `CUDNN_POOLING_MAX`를 사용할 때
- `cudnnSpatialTfSamplerBackward`
- `cudnnCTCLoss`와 `cudnnCTCLoss_v8`에 `CUDNN_CTC_LOSS_ALGO_NON_DETERMINSTIC`을 사용할 때

서로 다른 아키텍처에서의 cuDNN 루틴은 bit-wise reproducibility를 보장하지 않는다.

<br>

# Scaling Parameters

[`cudnnConvolutionForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward)와 같은 많은 cuDNN 루틴들은 scaling factor `alpha`와 `beta`에 대한 host memory 포인터를 받는다. Scaling factor는 계산된 값과 이전 결과 텐서의 값과 함께 사용되어 아래와 같이 계산하는데 사용된다.
```
dstValue = alpha * computedValue + beta * priorDstValue
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZoR3U%2Fbtsca1eMS7G%2FreyO5nyMK7lA0vlL5pT3OK%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

> `dstValue`는 read가 먼저 수행되고나서 write된다.

`beta`가 0이라면, 결과는 read되지 않으며 초기화되지 않은 값(`NaN` 포함)일 수 있다.

`alpha`와 `beta`는 host memory 포인터를 사용하여 전달되며, 이들의 storage data type은 다음과 같다.

- `float` for HALF and FLOAT tensors
- `double` for DOUBLE tensors

### Type Conversion

Data input `x`, filter input `w`, output `y`가 모두 INT8일 때, [`cudnnConvolutionBiasActivationForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward) 함수는 아래 그림과 같이 type conversion을 수행한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FTu3AI%2Fbtsb6ZBtmgg%2FrWfQeJmH3JbGd912KetZz0%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

<br>

# cuDNN API Compatibility

cuDNN 7부터 patch release와 minor release의 binary compatibility가 아래와 같이 유지된다.

- 모든 patch release `x.y.z`는 다른 patch release `x.y.w`(same major and minor version, but having `w!=z`)에 대해 빌드된 어플리케이션과 forward/backward-compatible 된다.
- cuDNN minor release는 동일하거나 이전 minor release에 대해 빌드된 어플리케이션과 backward-compatible 된다. 즉, `x.y`는 `x.z`에서 빌드된 어플리케이션과 호환 가능하다 (`z<=y`).
- cuDNN `x.z`로 컴파일된 어플리케이션은 `x.y` (`z>y`)에서 동작하지 않을 수 있다.

<br>

# Deprecation Policy

cuDNN version 8부터 새로운 API deprecation 정책을 도입했다.

이전 정책에서는 API 업데이트를 완료하기 위해 3번의 major release가 필요했다. 이 과정에서 원래 함수 이름은 legacy API로 먼저 할당되고, 그런 다음 수정된 API에 할당되었다. 그래서 새로운 API 버전으로 마이그레이션하려는 사용자는 자신의 코드를 두 번 업데이트해야 했다. 첫 번째 업데이트에서는 원래 이름인 `foo()`를 `foo_vN()`으로 변경해야 했는데, 여기서 `N`은 cuDNN의 새로운 major 버전이다. 그런 다음 두 번째 업데이트에서 `foo_vN()`을 다시 `foo()`로 변경해야 했다. 이러한 과정은 많은 기능이 업그레이드되는 경우 코드 유지 관리를 어렵게 만든다.

cuDNN version 8부터는 조금 더 간소화된 two-step deprecation 정책이 모든 API 변경에 적용된다. 8버전과 9버전의 major release를 예시로 해당 프로세스에 대해 살펴보자.

|cuDNN version|Explanation|
|:---|:--|
|Major release 8|업데이트되는 API는 `foo_v8()`의 형태로 도입된다. Deprecated API인 `foo()`는 이전 버전과의 호환성을 유지하기 위해 다음 major release까지 변경되지 않은 상태를 유지한다.|
|Major release 9|Deprecated API인 `foo()`가 영구적으로 제거되며, `foo()`라는 이름은 재사용되지 않는다. 대신 `foo()` 호출을 `foo_v8()`이 대체한다.|

기존 API를 업데이트해야 하는 경우, 기존 이름에 `_v` 태그가 major 버전과 함께 도입된다. 다음 major release에서는 deprecated function이 제거되고 제거된 API의 이름은 다시 사용되지 않는다. Brand-new API는 `_v` 태그없이 처음에 도입되었다.

수정된 deprecation scheme를 통해 legacy API는 한 번의 major release를 통해 폐기할 수 있다. 사용자는 다음 major release를 사용하여 변경없이 기존 코드를 컴파일할 수 있다. 그 다음 major release가 도입되면 이전 버전과의 호환성이 종료되므로 코드 수정이 필요하다.

업데이트된 함수 이름에는 cuDNN 버전 정보를 포함하도록 되어 있어서 API 변경 사항을 추적하고 문서화하기 더 쉽다.

새로운 deprecation 정책은 이전 cuDNN 릴리즈에서 보류 중인 API 변경 사항에도 적용된다. 예를 들어, 이전 정책에 따르면 `cudnnSetRNNDescriptor_v6()`는 cuDNN 버전 8에서 제거되어야 하며, 동일한 파라미터와 동작을 가진 `cudnnSetRNNDescriptor()`가 유지되어야 한다. 대신, 새로운 deprecation 정책이 이 경우에도 적용이 되며, 버전 태그가 지정된 함수가 유지된다.

Deprecated function의 프로토타입은 cuDNN version 8 헤더에 `CUDNN_DEPRECATED` 매크로를 사용하여 추가된다. `-DCUDNN_WARN_DEPRECATED`가 컴파일러에 전달되면, 모든 deprecated function call은 아래컴파일러 경고를 내보낸다.
```
warning: ‘cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t, cudnnMathType_t)’ is deprecated [-Wdeprecated-declarations]
```
or
```
warning C4996: 'cudnnSetRNNMatrixMathType': was declared deprecated
```

<br>

# GPU And Driver Requirements

OS, CUDA, CUDA Driver, NVIDIA Hardware의 호환성은 [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)를 참조

<br>

# Convolutions

cuDNN에는 아래의 Convolution 함수들이 있다.

- [`cudnnConvolutionBackwardData()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData)
- [`cudnnConvolutionBiasActivationForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward)
- [`cudnnConvolutionForward()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward)
- [`cudnnConvolutionBackwardBias()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardBias)
- [`cudnnConvolutionBackwardFilter()`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter)

해당 섹션에서는 `cudnnConvolutionForward()`에서 구현된 다양한 convolution 공식들, grouped convolution 등에 대해서 설명하고 있다. 자세한 내용은 공식 문서의 [Convolutions](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#convolutions)를 참조 바람.

<br>

# Environment Variables

몇 가지 환경 변수를 통해서 cuDNN 동작에 영향을 줄 수 있다. cuDNN에서 공식적으로 지원하는 환경변수는 다음과 같다.

- `NVIDIA_TF32_OVERRIDE` ([`cudnnMathType_t`](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t) 참조)
- `NVIDIA_LICENSE_FILE`
- `CUDNN_LOGDEST_DBG`
- `CUDNN_LOGINFO_DBG`
- `CUDNN_LOGWARD_DBG`
- `CUDNN_LOGERR_DBG`

해당 환경변수들에 인한 동작은 [API References](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)에서 찾아볼 수 있다.

<br>

# References
- [NVIDIA cuDNN Documentation: Odds and Ends](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#misc)