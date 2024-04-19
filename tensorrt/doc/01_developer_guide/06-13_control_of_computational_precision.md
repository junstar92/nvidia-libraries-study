# Table of Contents

- [Table of Contents](#table-of-contents)
- [Control of Computational Precision](#control-of-computational-precision)
- [References](#references)

<br>

> Updated in TensorRT 10.0.0

# Control of Computational Precision

때때로, 한 연산의 입출력 정밀도를 설정하는 것 외에도 연산의 내부 정밀도를 제어할 필요가 있을 수 있다. 기본적으로 TensorRT는 layer input type과 global performance 고려를 기반으로 연산 정밀도를 선택한다.

TensorRT에는 연산 정밀도를 조절하는 기능을 가진 두 개의 레이어가 있다.

- `INormalizationLayer` - `setPrecision` 메소드를 통해 accumulation의 정밀도를 제어할 수 있다. 기본적으로 overflow error를 피하기 위해 FP32로 누적하며, 심지어 mixed precision mode에서도 FP32로 누적한다. 이 메소드를 사용하면 FP16 정밀도로 누적시킬 수 있다.
- `IMatrixMultiplyLayer` - 이 레이어는 기본적으로 input types과 performance considerations를 기반으로 accumulation precision을 선택한다 하지만 accumulation type은 입력 타입만큼의 범위를 갖도록 보장된다. Strongly-typed mode를 사용하는 경우에는 입력을 FP32로 캐스팅하여 FP16 GEMM에 FP32 precision을 사용하도록 할 수 있다. TensorRT는 이 패턴을 인식하여 FP16 to FP32 cast를 GEMM과 fusion하여 FP16 input 및 FP32 accumulation의 단일 커널을 만든다.

# References

- [NVIDIA TensorRT Documentation: Control of Computational Precision](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#control-comput-precision)