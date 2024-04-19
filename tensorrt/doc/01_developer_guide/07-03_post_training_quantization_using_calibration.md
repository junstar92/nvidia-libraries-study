# Table of Contents

- [Table of Contents](#table-of-contents)
- [Post-Training Quantization Using Calibration](#post-training-quantization-using-calibration)
- [INT8 Calibration Using C++](#int8-calibration-using-c)
- [Calibration Using Python](#calibration-using-python)
- [Quantization Noise Reduction](#quantization-noise-reduction)
- [References](#references)

# Post-Training Quantization Using Calibration

> [TensorRT 10.0] **INT8 quantization만 가능하다.**

Post-Training Quantization (PTQ)에서 TensorRT는 네트워크의 각 텐서에 대한 scale 값을 계산한다. 이러한 프로세스를 _calibration_ 이라고 부르며, calibration에는 각 activation 텐서에 대한 통계(statistics) 정보를 수집하기 위해서 네트워크에 대한 대표 입력 데이터(representative input data)가 필요하다. 즉, TensorRT는 calibration 프로세스를 위해 네트워크를 실행하는데 실제 입력 데이터가 필요하다는 것이다.

필요한 입력 데이터의 양은 application-dependent이다. 하지만 실험적으로 ImageNet classification network에 대해 calibration을 수행할 때, 대략 500개의 이미지면 충분하다고 한다.

Activation 텐서에 대한 통계가 주어지면, 최상의 scale value를 결정하는 것은 정확한 과학(exact science)이 아니다. Scale value는 양자화된 값에서 두 가지 에러, 즉, **discretization error**와 **truncation error**와의 밸런스를 유지해야 한다.

> Discretization error는 양자회된 값이 표현하는 범위가 커질수록 증가하며, truncation error는 표현 가능한 범위를 벗어나면 해당 범위의 경계로 값이 고정되기 때문에 발생한다.

따라서, TensorRT는 다양한 방식으로 scale을 계산하는 여러 calibrator를 제공한다. 초기 calibrator는 calibration을 수행하기 전에 불필요한 텐서를 최적화하기 위해 GPU에 대한 layer fusion을 수행했다. 하지만 이는 fusion pattern이 다를 수 있는 DLA을 사용할 때 문제가 될 수 있으며, 이러한 fusion은 `kCALIBRATE_BEFORE_FUSION` quantization flag를 사용하여 무시할 수 있다.

Calibration batch size는 `IInt8EntropyCalibrator2`와 `IInt8EntropyCalibrator`의 truncation error에 영향을 줄 수 있다. 예를 들어, calibration data에서 여러 작은 batches를 사용하여 calibration을 수행하면 histogram resolusion이 감소하고 scale value가 별로 좋지 않을 수 있다. 매 calibration step에서 TensorRT는 각 activation 텐서에 대한 히스토그램 분포를 업데이트한다. 만약 activation 텐서에서 히스토그램의 최댓값보다 큰 값을 만나면 히스토그램의 볌위는 2의 거듭제곱으로 증가하게 된다. 이러한 접근 방식은 마지막 스텝에서 히스토그램 reallocation이 발생하지 않는 한 잘 동작하며, 최종 히스토그램에서 절반의 bins가 비어 있게 된다. 이러한 히스토그램은 잘못된 calibration scales를 생성할 수 있다. 또한, calibration을 배치 순서에 민감하게 만든다. 즉, 배치의 순서가 다르면 히스토그램의 크기가 다른 지점에서 증가하고 약간 다른 calibration scales가 생성될 수 있다. 이러한 문제를 방지하려면, 가능한 한 하나의 큰 단일 배치로 calibration을 수행하고, calibration batches가 원래의 데이터의 분포와 유사하도록 해야 한다.

> 정리하면, calibration batches의 크기가 작으면 히스토그램의 resolution이 감소하고, scale value는 좋지 않을 수 있다. 또한, 배치의 순서에 따라 calibration 결과가 달라질 수 있다. 이러한 문제를 방지하려면, 가능한 한 큰 단일 배치로 calibration을 수행해야 한다.

- `IInt8EntropyCalibrator2`
  
  Entropy calibration은 텐서의 scale factor를 선택하여 quantized tensor의 information-theoretic content를 최적화하고, 일반적으로 분포 내 이상값을 억제한다. 현재 권장하는 entropy calibrator이며 DLA에서는 이를 사용해야 한다. Calibration은 기본적으로 layer fusion 이전에 발생한다. Calibration batch size는 최종 결과에 영향을 줄 수 있다. 특히, CNN 기반 네트워크에서 권장되는 calibrator이다.

- `IInt8MinMaxCalibrator`
  
  Activation 분포의 전체 범위를 사용하여 scale factor를 결정한다. NLP tasks에서 잘 동작하는 것으로 보이며, calibration은 기본적으로 layer fusion 이전에 발생한다. NVIDIA BERT(an optimized verison of [Google's official implementation](https://github.com/google-research/bert))와 같은 네트워크에서 권장된다.

- `IInt8EntropyCalibrator`
  
  Original entropy calibrator이다. LegacyCalibrator보다 사용하기에 덜 복잡하고 일반적으로 더 좋은 결과를 생성한다. 마찬가지로 calibration batch size는 최종 결과에 영향을 줄 수 있다. Calibration은 기본적으로 layer fusion 이후에 발생한다.

- `IInt8LegacyCalibrator`
  
  이는 TensorRT 2.0 EA와의 호환성을 위한 것이다. 이 calibrator는 user parameterization이 필요하고, 만약 다른 calibrator의 결과가 좋지 않는 경우 fallback option으로 제공된다. Calibration은 기본적으로 layer fusion 이후에 발생한다. 이 calibrator를 커스터마이징하여 percentile max를 구현할 수 있는데, 예를 들어, 99.99% percentile max는 NVIDIA BERT 및 NeMo ASR model QuartzNet에서 최고의 정확도를 갖는 것으로 관측된다.

INT8 엔진을 빌드할 때, builder는 아래의 과정을 수행한다.

1. Build a 32-bit engine, run it on the calibration set, and record a histogram for each tensor of the distribution of activation values.
2. Build from the histograms a calibration table providing a scale value for each tensor.
3. Build the INT8 engine from the calibration table and the network definition.

Calibration은 느릴 수 있다. 그러므로 step 2의 결과(calibration table)을 캐싱하고 재사용하면 도움이 된다. 특히 같은 네트워크를 여러 번 빌드할 때 유용하며, 모든 calibrators에서 캐싱 기능이 지원된다.

Calibration을 수행하기 전에 TensorRT는 calibration implementation을 쿼리하여 cached table에 대한 액세스 권한이 있는지 확인한다. 권한이 있다면, 곧바로 step 3을 진행한다. Cached data는 pointer와 length로 전달되며, 샘플 calibration data는 [link](https://github.com/NVIDIA/TensorRT/tree/release/8.4/samples/sampleINT8#calibration-file)에서 볼 수 있다.

Calibration cache data는 layer fusion 이전에 calibration이 수행되는 한 여러 device에서 호환 가능하다. 특히 `IInt8EntropyCalibrator2` 또는 `IInt8MinMaxCalibrator`를 사용하거나 `QuantizationFlag::kCALIBRATE_BEFORE_FUSION`이 설정된 경우, portable 하다. 예를 들어, 별도의 GPU가 있는 시스템에서 calibration table을 구성한 다음, 임베디드 플랫폼에서 재사용하여 프로세스를 단수화할 수 있다. Fusion은 플랫폼이나 GPU 장치 간에 동일하다고 보장되지 않기 때문에 layer fusion 이후 calibration을 수행하면 portable calibration cache가 생성되지 않을 수 있다. Calibration cache는 일반적으로 TensorRT 릴리즈 간에는 호환되지 않는다.

<br>

TensorRT는 activation뿐만 아니라 weights도 양자화해야 한다. Weight 텐서에서 발견한 maximum absolute values를 사용하여 계산한 quantization scale을 사용하는 symmetric quantization을 수행한다. Convolution, deconvolution, fully connected weights에서 scales은 채널 별로 갖는다.

> INT8 I/O를 사용하도록 빌드가 구성되더라도, TensorRT는 calibration data가 여전히 FP32 타입이라고 예상한다. INT8 I/O calibraiton data를 FP32 정밀도로 캐스팅하여 FP32 calibraiton data를 생성할 수 있다. 이때, 값의 범위가 [-128.0F, 127.0F]가 되도록 해야 하며, 따라서, 어떠한 precision loss없이 INT8 data로 변환될 수 있다.

INT8 calibration은 dynamic range APIs와 함께 사용할 수 있다. Dynamic range를 수동으로 설정하면 INT8 calibration으로부터 생성된 dynamic range는 무시된다.

> Calibration은 **deterministic** 이다. 즉, 동일한 순서의 동일한 입력, 동일한 device에서는 동일한 scale이 생성된다. 이러한 환경에서 calibration cache 데이터는 bit 단위로 동일하다. 다른 device, 다른 batch size, 다른 calibraiton data에서는 bit 단위로 동일하다고 보장되지 않는다.

<br>

# INT8 Calibration Using C++

TensorRT에 calibration data를 제공하려면, `IInt8Calibrator` 인터페이스를 구현해야 한다.

빌더는 아래의 순서로 calibrator를 호출한다.

- 먼저, batch size에 대한 인터페이스를 쿼리하고 `getBatchSize()`를 호출하여 기대하는 input batch의 크기를 결정한다.
- 그런 다음, `getBatch()`를 반복적으로 호출하여 입력 배치를 얻는다. 이 입력 배치의 크기는 `getBatchSize()`로 얻은 크기와 정확히 같아야 한다. 더 이상 얻을 입력 배치가 없다면 `getBatch()`는 `false`를 리턴해야 한다.

Calibrator 구현한 뒤, 다음과 같이 builder를 구성할 수 있다.
```c++
config->setInt8Calibrator(calibrator.get());
```

Calibration table을 캐싱하려면, `writeCalibrationCache()`과 `readCalibrationCache()` 메소드를 구현해야 한다.

<br>

# Calibration Using Python

Python에서의 calibraiton은 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_python)를 참조

<br>

# Quantization Noise Reduction

Implicit quantization의 네트워크의 경우, TensorRT는 INT8 구현을 사용할 수 있더라도 네트워크 출력 근처의 일부 레이어를 FP32로 실행하도록 하여 출력의 quantization noise를 줄이려고 시도한다.

휴리스틱은 INT8 양자화가 여러 양자화된 값의 합으로 부드럽게 처리되도록 보장한다. 네트워크의 출력에 도달하기 전에 "smoothing layer"로 간주되는 레이어는 convolution, deconvolution, fully connected, 또는 matrix multiplication이다. 예를 들어, 네트워크가 일련의 (convolution + activation + shuffle)의 subgraph로 구성되고, 네트워크 출력의 타입이 FP32인 경우, 마지막 convolution에서는 INT8이 허용되고 더 빠르더라도 FP32 precision으로 출력한다.

휴리스틱은 다음의 시나리오에서는 적용되지 않는다.

- The network output has type INT8.
- An operation on the path (inclusively) from the last smoothing layer to the output is considered by `ILayer::setOutputType` or `ILayer::setPrecision` to output INT8.
- There is no smoothing layer with a path to the output, or said that path has an intervening plugin layer.
- The network uses explicit quantization.

<br>

> NVIDIA에서 PTQ는 `Implicit Quantization`에 해당한다. [Explicit Versus Implicit Quantization](/tensorrt/doc/01_developer_guide/07-01_introducing_to_quantization.md#explicit-versus-implicit-quantization)에서 간략하게 설명하고 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Post-Training Quantization Using Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c)