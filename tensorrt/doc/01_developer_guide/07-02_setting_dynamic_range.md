# Table of Contents

- [Table of Contents](#table-of-contents)
- [Setting Dynamic Range](#setting-dynamic-range)
- [References](#references)

# Setting Dynamic Range

TensorRT는 *dynamic range* 를 직접 설정하는 API를 제공한다. 이 API는 TensorRT 외부에서 계산된 implicit quantization을 지원하기 위해 지원된다.

> Dynamic range는 양자화된 텐서로 표현되는 범위이다. 

이 API는 minimum/maximum 값을 사용하여 텐서의 dynamic range를 설정한다. TensorRT는 현재 symmetric range만을 지원하며, scale은 `max(abs(min_float), abs(max_float))`로 계산된다. `abs(min_float) != abs(max_float)` 일 때, TensorRT는 설정된 것보다 더 큰 dynamic-range를 사용하므로 rounding error가 증가할 수 있다.

Dynamic range는 INT8로 실행되는 연산의 모든 floating-point inputs/outputs에 필요하다.

C++에서는 다음과 같이 텐서의 dynamic range를 설정할 수 있다.
```c++
tensor->setDynamicRange(min_float, max_float);
```

> [sampleINT8API](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleINT8API)에서 관련된 API를 사용하는 방법을 잘 보여준다.

# References

- [NVIDIA TensorRT Documentation: Setting Dynamic Range](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#set-dynamic-range)
- [TensorRT Sample: sampleINT8API](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleINT8API)