# Table of Contents

- [Table of Contents](#table-of-contents)
- [Optimizing for Tensor Cores](#optimizing-for-tensor-cores)
- [References](#references)

<br>

# Optimizing for Tensor Cores

텐서 코어(Tensor Core)는 NVIDIA GPU에서 고성능의 추론을 제공하는 핵심 기술이다. TensorRT에서 텐서 코어는 `MatrixMultiply`, `FullyConnected`, `Convolution` 및 `Deconvolution`과 같은 모든 compute-intensive 레이어에서 지원된다.

텐서 코어 레이어는 I/O 텐서 차원이 특정 minimum granularity에 정렬되면 더 좋은 성능을 달성하는 경향이 있다.

- Convolution과 Deconvolution 레이어에서의 alignment requirement는 I/O channel dimension에 있다.
- MatrixMulitply와 FullyConnected 레이어에서의 alignment requirement는 `M x K` 행렬과 `K x N` 행렬 곱셈에서 `K`와 `N` 차원에 있다.

아래 테이블은 더 나은 텐서 코어 성능을 위해 제안되는 텐서 차원의 alignment를 나열한다.
|Tensor Core Operation Type|Suggested Tensor Dimension Alignment in Elements|
|--|--|
|TF32|4|
|FP16|8 for dense math, 16 for sparse math|
|INT8|32|

이러한 요구 사항이 충족되는 않는 상황에서 텐서 코어 구현을 사용할 때, TensorRT는 계산 또는 메모리 트래픽을 증가시키지 않고 모델에 추가 용량을 허용하는 대신 텐서의 차원을 가장 가까운 alignment 배수로 pad를 채운다.

> TensorRT는 항상 가장 빠른 구현을 사용하므로 텐서 코어 구현 사용이 가능하더라도 해당 구현이 선택되지 않을 수 있다.

<br>

텐서 코어가 레이어에서 사용되는지 확인하려면 `--gpu-metrics-device all` 플래그와 함께 Nsight Systems를 실행하면 확인할 수 있다. 텐서 코어의 usage rate는 프로파일링 결과의 **SM Instructions/Tensor Active** 행에서 확인할 수 있다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/tc-usage.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>


<br>

# References

- [NVIDIA TensorRT Documentation: Optimizing for Tensor Cores](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-layer)