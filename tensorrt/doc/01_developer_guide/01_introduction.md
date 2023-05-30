# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Samples](#samples)
- [Complementary GPU Features](#complementary-gpu-features)
- [Complementary Software](#complementary-software)
- [ONNX](#onnx)
- [Code Analysis Tools](#code-analysis-tools)
- [API Versioning](#api-versioning)
- [Deprecation Policy](#deprecation-policy)
- [Hardware Support Lifetime](#hardware-support-lifetime)
- [References](#references)

<br>

# Introduction

NVIDIA TensorRT는 고성능의 머신러닝 추론(inference)를 위한 SDK이다. TensorFlow, PyTorch, MXNet과 같은 학습 프레임워크를 보완하도록 설계되었으며, 미리 학습된 네트워크를 NVIDIA 하드웨어에서 빠르고 효율적으로 실행하는데 초점을 둔다.

<br>

# Samples

[NVIDIA TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)에 다양한 샘플 코드가 있다. 임베디드 어플리케이션을 위한 샘플은 [link](https://github.com/dusty-nv/jetson-inference)에서 확인할 수 있다.

<br>

# Complementary GPU Features

<br>

# Complementary Software

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/#nvidia-dali-documentation)
- TensorFlow-TensorRT ([TF-TRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html))
- Torch-TensorRT ([Torch-TRT](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/))
- [TensorFlow-Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)
- [PyTorch Quantization Toolkit](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)
- [PyTorch Automatic SParsity (ASP)](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)

<br>

# ONNX

[ONNX](https://onnx.ai/) interchange format를 통해 프레임워크로부터 학습된 모델을 사용하여 TensorRT 엔진으로 변환할 수 있다. TensorRT는 ONNX parser 라이브러리와 함께 제공되며, 이를 통해 model import를 지원한다.

Github에서 제공되는 ONNX 버전은 TensorRT와 함께 제공되는 버전보다 최신 opset을 지원할 수 있다. 지원되는 opset 및 연산자에 대한 최신 정보는 [ONNX-TensorRT operator support matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)에서 확인할 수 있다.

PyTorch는 기본적으로 ONNX export를 지원한다. TensorFlow의 경우, [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) 사용을 권장한다.

각 프레임워크에서 학습된 모델을 ONNX 포맷으로 export한 후에는 [Polygraphy](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#polygraphy-ovr)를 통해 지속적인 폴딩(folding)을 실행하는 것이 좋다. 이를 통해 ONNX parser에서의 TensorRT 변환 문제를 해결하고, 일반적으로 워크플로우를 단순화할 수 있다. 이에 대한 [example](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants)를 통해 자세한 내용을 살펴볼 수 있다.

경우에 따라서는 subgraph를 플러그인(plugins)으로 교체하거나 지원되지 않는 연산을 다른 연산으로 재구현하기 위해 ONNX 모델을 추가로 수정해야할 수도 있다. 이 프로세스를 쉽게 하기 위해서 [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)을 사용할 수 있다.

<br>

# Code Analysis Tools

[Code Analysis Tools](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#code-analysis-tools-ovr) 참조

<br>

# API Versioning

TensorRT 버전 (MAJOR.MINOR.PATCH)는 Semantic Versioning 2.0.0을 따른다.

- MAJOR version when making incompatible API or ABI changes
- MINOR version when adding functionality in a backward compatible manner
- PATCH version when marking backward compatible bug fixes

Plan 파일이나 timing caches를 재사용하기 위해서는 모든 버전이 일치해야 한다. Calibration caches의 경우, 일반적으로 major 버전 내에서 재사용할 수 있지만 호환성을 보장하지는 않는다.

# Deprecation Policy

Deprecation 정책에 관련해서는 문서([link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#deprecation)) 참조

<br>

# Hardware Support Lifetime

TensorRT 8.5.3은 NVIDIA Kepler(SM 3.X)와 Maxwell(SM 5.X)를 지원하는 마지막 릴리즈이다. 이후 릴리즈에서는 Kepler와 Maxwell은 더 이상 지원되지 않는다. NVIDIA Pascal(SM 6.X)는 TensorRT 8.6에서 더 이상 지원되지 않는다.

<br>

# References

- [NVIDIA TensorRT Documentation: Introduction](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#overview)
- [NVIDIA TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
- [Gihub: jetson-inference](https://github.com/dusty-nv/jetson-inference)
- [ONNX](https://onnx.ai/)
- [ONNX-TensorRT operator support matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)