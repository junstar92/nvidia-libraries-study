# Table of Contents

- [Table of Contents](#table-of-contents)
- [I/O Formats](#io-formats)
- [References](#references)

<br>

# I/O Formats

TensorRT는 다양한 data format들을 사용하여 네트워크를 최적화할 수 있다. TensorRT와 client application 간 데이터의 효율적인 전달을 위해서 네트워크의 I/O boundaries, 즉, 네트워크의 input 또는 output으로 마크된 텐서들의 기본 데이터 타입이 노출된다. 다른 텐서들은 전체적으로 가장 빠른 실행을 위해 TensorRT가 해당 텐서들의 포맷을 선택하며 성능을 향상시키기 위해 reformats를 삽입할 수도 있다.

TensorRT 전후 연산에 가장 효율적인 포맷의 조합으로 I/O formats을 프로파일링하여 최적의 data pipeline을 조합할 수 있다.

I/O formats을 지정하기 위해서 하나 또는 하나 이상의 포맷을 비트마스크 형태로 지정하면 된다.

아래 예제 코드는 input 텐서 포맷을 `TensorFormat::kHWC8`로 지정한다. 참고로 이 포맷은 `DataType::kHALF`에서만 동작한다. 따라서 이에 맞는 데이터 타입도 설정해야 한다.
```c++
auto formats = 1U << TensorFormat::kHWC8;
network->getInput(0)->setAllowedFormats(formats);
network->getInput(0)->setType(DataType::HALF);
```

> 네트워크의 input/output이 아닌 텐서에 대한 `setAllowedFormats()` 또는 `setType()` 호출은 아무런 효과도 없으며 TensorRT에 의해서 무시된다.

Builder configuration 플래그 `DIRECT_IO`를 설정하여 TensorRT가 네트워크 경계에서 reformatting을 삽입하는 것을 피하도록 할 수 있다. 이 플래그는 일반적으로 아래의 두 가지 이유로 **비생산적**이다.

- 생성된 엔진이 TensorRT가 reformatting을 삽입하도록 허용하는 것보다 더 느릴 수 있다. Reformatting이 낭비되는 작업처럼 보이지만, 가장 효율적인 커널과 결합할 수 있다.
- 이러한 reformatting없이 엔진을 빌드할 수 없다면 빌드가 실패한다. 플랫폼마다 커널에서 지원되는 포맷으로 인해서, 이 실패는 오직 몇몇 타겟 플랫폼에서만 발생할 수 있다.

이 플래그는 reformatting을 위해 GPU로 fallback하지 않고 DLA에서만 실행되는 엔진을 빌드할 때와 같이 I/O boundaries에서 발생하는 reformatting을 완전히 제어하고자 하는 사용자를 위해 존재한다.

> [sampleIOFormats](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleIOFormats)에서 어떻게 I/O 포맷을 제어할 수 있는지 보여준다.

아래 표는 지원되는 포맷을 보여준다.

|Format|`kINT32`|`kFLOAT`|`kHALF`|`kINT8`|
|------|--------|--------|-------|-------|
|`kLINEAR`|Only for GPU|Supported|Supported|Supported|
|`kCHW2`|Not Applicable|Not Applicable|Only for GPU|Not Applicable|
|`kCHW4`|Not Applicable|Not Applicable|Supported|Supported|
|`kHWC8`|Not Applicable|Not Applicable|Only for GPU|Not Applicable|
|`kCHW16`|Not Applicable|Not Applicable|Supported|Not Applicable|
|`kCHW32`|Not Applicable|Only for GPU|Only for GPU|Supported
|`kDHWC8`|Not Applicable|Not Applicable|Only for GPU|Not Applicable|
|`kCDHW32`|Not Applicable|Not Applicable|Only for GPU|Only for GPU|
|`kHWC`|Not Applicable|Only for GPU|Not Applicable|Not Applicable|
|`kDLA_LINEAR`|Not Applicable|Not Applicable|Only for DLA|Only for DLA|
|`kDLA_HWC4`|Not Applicable|Not Applicable|Only for DLA|Only for DLA|
|`kHWC16`|Not Applicable|Not Applicable|Only for NVIDIA Ampere GPUs and later|Not Applicable|
|`kDHWC`|Not Applicable|Only for GPU|Not Applicable|Not Applicable|

Vectorized formats의 경우, 채널 차원은 반드시 벡터 사이즈 배수로 zero-padding해야 한다. 예를 들어, 만약 바인딩된 input의 차원이 [16,3,244,244]이고, 타입은 `kHALF`, `kHWC8` 포맷이라면, 비록 `engine->getBindingDimension()` API로 리턴한 input 텐서의 차원이 [16,3,224,244]라도 실제 버퍼에 필요한 크기는 `16*8*224*224*sizeof(half)` bytes이다. 여기서 사용되지 않는 패딩 처리된 부분(`C=3,4,...,7`)은 반드시 0으로 채워져야 한다.

> 포맷에 대한 자세한 내용은 [Data Format Descriptions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc)에서 살펴볼 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: I/O Formats](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors)
- [TensorRT Samples: sampleIOFormats](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleIOFormats)
- [NVIDIA TensorRT Documentation: Data Format Descriptions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc)