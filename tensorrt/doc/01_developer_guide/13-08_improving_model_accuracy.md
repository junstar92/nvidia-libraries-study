# Table of Contents

- [Table of Contents](#table-of-contents)
- [Improving Model Accuracy](#improving-model-accuracy)
- [References](#references)

<br>

# Improving Model Accuracy

TensorRT는 build configuration에 따라서 레이어를 FP32, FP16, INT8 정밀도로 실행할 수 있다. 기본적으로 TensorRT는 최적의 성능을 제공하는 정밀도로 레이어를 실행하도록 선택한다. 이로 인해서 정확도가 떨어질 수는 있다. 일반적으로 더 높은 정밀도로 레이어를 실행하면 정확도는 향상되지만 약간의 성능 저하가 있다.

아래는 모델의 정확도를 향상시킬 수 있는 여러 단계를 보여준다.

1. Validate layer outputs
   - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)를 사용하여 결과를 덤프하고 NaNs 또는 Infs가 없는지 검증한다. `--validate` 옵션으로 이를 검증할 수 있다. 또한 출력을 ONNX runtime 등을 사용하여 얻은 결과(golden values)와 비교한다.
   - `FP16`인 경우, 중간 레이어의 출력이 오버플로우/언더플로우 없이 FP16 정밀도로 표현될 수 있도록 retraining이 필요할 수 있다.
   - `INT8`인 경우, 보다 대표적인 calibration dataset으로 recalibration 하는 것을 고려하라. 만약 PyTorch 모델을 사용한다면, PTQ 뿐만 아니라 QAT를 위한 PyTorch용 NVIDIA Qunatization Toolkit을 제공한다. 두 가지 quantization 방법을 모두 시도하고 더 정확한 방법을 선택할 수 있다.
2. Manipulate layer precision:
   - 때때로 특정 정밀도로 레이어를 실행할 때 부정확한 결과를 얻을 수 있다. 이는 고유한 레이어의 제약 조건, 모델의 제약 조건 또는 TensorRT의 버그 때문일 수 있다. (예를 들어, `LayerNorm`의 출력은 `INT8`이 아니어야 하거나 출력이 분기되어서 정확도가 떨어지는 경우 등)
   - Layer의 execution precision과 output precision을 제어할 수 있다.
   - [debug precision](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools/debug) tool(in Polygraphy)이 도움이 될 수 있다 (높은 정밀도로 실행되는 레이어를 자동으로 찾아줌).
3. Use an [Algorithm Selection and Reproducible Builds](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select) to disable flaky tactics
   - build+run과 build+run 간의 정확도 변경은 레이어에 대해 bad tactic 선택 때문일 수 있다.
   - 최적의 실행과 좋지 않은 실행으로부터 tactics를 덤프하기 위해 algorithm selector를 사용한다. Algorithm selector를 구성하여 오직 tactics의 일부 서브셋만 허용하도록 할 수 있다. 즉, good run의 tactics만 허용하는 등의 작업이 가능하다.
   - Polygraphy의 [automate](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics)를 통해 위 프로세스를 자동화할 수 있다.

run-to-run variation에서의 정확도는 변경되면 안된다. 엔진이 특정 GPU에 대해 빌드되면, 여러 실행에서 비트 단위로 일치되도록 출력해야 한다. 그렇지 않는 경우에는 버그에 해당한다.

<br>

# References

- [NVIDIA TensorRT Documentation: Improving Model Accuracy](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#model-accuracy)
- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)