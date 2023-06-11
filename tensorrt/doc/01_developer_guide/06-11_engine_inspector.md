# Table of Contents

- [Table of Contents](#table-of-contents)
- [Engine Inspector](#engine-inspector)
- [References](#references)

<br>

# Engine Inspector

TensorRT에서는 `IEngineInspector` API를 통해 엔진의 정보를 알아낼 수 있다. 역직렬화(deserialization)된 엔진을 통해 `createEngineInspector()`를 호출하여 engine inspector를 생성할 수 있고, 생성한 inspector를 통해 `getLayerInformation()` 또는 `getEngineInformation()` API를 호출하여 특정 레이어의 정보나 엔진 전체 정보를 얻을 수 있다. 사용법은 다음과 같다.
```c++
auto inspector = std::unique_ptr<IEngineInspector>(engine->createEngineInspector());
inspector->setExecutionContext(context); // optional
std::cout << inspector->getLayerInformation(0, LayerInformationFormat::kJSON); // Print the information of the first layer in the engine.
std::cout << inspector->getEngineInformation(LayerInformationFormat::kJSON); // Print the information of the entire engine.
```

엔진 또는 레이어를 조사하여 출력되는 정보의 수준(detail)은 엔진을 빌드할 때 설정되는 `ProfileVerbosity` build config에 따라 다르다. `ProfileVerbosity`가 `kLAYER_NAMES_ONLY`로 설정되면, 오직 레이어의 이름만 출력된다. 만약 `kNONE`으로 설정된다면 어떠한 정보도 출력되지 않고, `kDETAILED`로 설정되면 세부 정보가 모두 출력된다.

아래는 `kLAYER_NAMES_ONLY`로 설정되었을 때의 출력 예시이다.
```
"node_of_gpu_0/res4_0_branch2a_1 + node_of_gpu_0/res4_0_branch2a_bn_1 + node_of_gpu_0/res4_0_branch2a_bn_2"
```

`kDEATILED`로 설정하면 아래와 같은 출력을 얻을 수 있다.
```
{
  "Name": "node_of_gpu_0/res4_0_branch2a_1 + node_of_gpu_0/res4_0_branch2a_bn_1 + node_of_gpu_0/res4_0_branch2a_bn_2",
  "LayerType": "CaskConvolution",
  "Inputs": [
  {
    "Name": "gpu_0/res3_3_branch2c_bn_3",
    "Dimensions": [16,512,28,28],
    "Format/Datatype": "Thirty-two wide channel vectorized row major Int8 format."
  }],
  "Outputs": [
  {
    "Name": "gpu_0/res4_0_branch2a_bn_2",
    "Dimensions": [16,256,28,28],
    "Format/Datatype": "Thirty-two wide channel vectorized row major Int8 format."
  }],
  "ParameterType": "Convolution",
  "Kernel": [1,1],
  "PaddingMode": "kEXPLICIT_ROUND_DOWN",
  "PrePadding": [0,0],
  "PostPadding": [0,0],
  "Stride": [1,1],
  "Dilation": [1,1],
  "OutMaps": 256,
  "Groups": 1,
  "Weights": {"Type": "Int8", "Count": 131072},
  "Bias": {"Type": "Float", "Count": 256},
  "AllowSparse": 0,
  "Activation": "RELU",
  "HasBias": 1,
  "HasReLU": 1,
  "TacticName": "sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize256x128x64_stage4_warpsize4x2x1_g1_tensor16x8x32_simple_t1r1s1_epifadd",
  "TacticValue": "0x11bde0e1d9f2f35d"
}
```

또한, dynamic shape로 엔진이 빌드될 때, 엔진 정보에서 dynamic dimensions는 `-1`로 출력되며, tensor format 정보는 inference phase에서 실제 shape에 따라 다르므로 출력되지 않는다. 구체적인 inference shape에 대한 엔진 정보를 얻고 싶다면, `IExecutionContext`을 생성하고 모든 입력 차원을 설정한 뒤, `inspector->setExecutionContext(context)`를 호출하면 된다. `context`가 설정된 후, `inspector`는 해당 `context`에서 지정된 shape에 대한 엔진 정보를 출력할 수 있다.

`trtexec` tool은 `--profileVerbosity`, `--dumpLayerInfo`, `--exportLayerInfo` 옵션을 제공하며, 이를 통해 주어진 엔진의 정보를 얻을 수 있다 ([trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) 참조).

현재, 오직 binding information과 layer information만 엔진 정보에 포함된다. 포함되는 정보는 다음과 같다.

- the dimensions of the intermediate tensors
- precisions
- formats
- tactic indices
- layer types
- layer parameters

이후 릴리즈에서 더 많은 정보가 engine inspector에 추가될 수도 있다. Output JSON의 key와 field에 대한 정보 또한 제공될 예정이다.

> next-generation graph optimizer에서 처리되는 몇몇 subgraphs는 아직 engine inspector에 통합되지 않았다. 따라서, 이러한 레이어에 대한 레이어 정보는 현재 보여지지 않는다.

<br>

# References

- [NVIDIA TensorRT Documentation: Engine Inspector](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector)