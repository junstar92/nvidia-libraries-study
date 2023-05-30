# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installing TensorRT](#installing-tensorrt)
- [The TensorRT Ecosystem](#the-tensorrt-ecosystem)
  - [Basic TensorRT Workflow](#basic-tensorrt-workflow)
  - [Conversion and Deployment Options](#conversion-and-deployment-options)
  - [Selecting the Correct Workflow](#selecting-the-correct-workflow)
- [Example Deployment Using ONNX](#example-deployment-using-onnx)
- [TF-TRT Framework Integration](#tf-trt-framework-integration)
- [ONNX Conversion and Deployment](#onnx-conversion-and-deployment)
  - [Exporting with ONNX](#exporting-with-onnx)
  - [Converting ONNX to a TensorRT Engine](#converting-onnx-to-a-tensorrt-engine)
  - [Deploying a TensorRT Engine to the Python Runtime API](#deploying-a-tensorrt-engine-to-the-python-runtime-api)
- [Using the TensorRT Runtime API](#using-the-tensorrt-runtime-api)
- [References](#references)

<br>

# Introduction

NVIDIA TensorRT는 고성능의 추론을 위해 학습된 딥러닝 모델을 최적화하는 SDK이다. TensorRT에는 학습되어 있는 딥러닝 모델을 최적화하는 딥러닝 inference optimizer와 실행을 위한 runtime을 포함한다.

Pytorch나 Tensorflow와 같은 프레임워크에서 딥러닝 모델을 학습한 뒤, TensorRT를 사용하면 더 높은 처리량과 더 짧은 latency로 모델을 실행시킬 수 있다.

Quick Start Guide에서는 아래 내용들을 다룬다.

- Basic installation
- Conversion
- Runtime options available in TensorRT

<br>

# Installing TensorRT

TensorRT 설치 방법은 [Installing TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#install)를 참조 바람.

<br>

# The TensorRT Ecosystem

TensorRT는 유연하다. 다양한 변환 및 배포 워크플로우를 다룰 수 있으며, 특정 use case와 문제 설정에 따라서 적합한 워크플로우가 달라질 수 있다.

TensorRT는 다양한 배포 옵션을 제공하지만, 모든 워크플로우에는 TensorRT가 엔진(engine)이라고 부르는 것으로 모델을 최적화하여 변환하는 작업이 포함된다. 모델에 대한 TensorRT 워크플로우를 빌드하려면 올바른 배포 옵션을 선택하고 엔진을 생성하기 위한 올바른 매개변수 조합을 선택해야 한다.

## Basic TensorRT Workflow

모델을 변환하고 배포하는 기본적인 과정은 다음과 같다.

1. Export The Model
2. Select A Batch Size
3. Select A Precision
4. Convert The Model
5. Deploy The Model

[Example Deployment Using ONNX](#example-deployment-using-onnx)에서는 ONNX를 사용하여 ResNet-50 모델을 변환하고 배포하는 방법을 보여준다.

## Conversion and Deployment Options

TensorRT Ecosystem은 두 부분으로 나뉜다.

1. TensorRT는 다양한 프레임워크에서 학습된 딥러닝 모델을 TensorRT 엔진으로 변환할 수 있음
2. TensorRT 엔진은 다양한 런타임(runtime)에서 실행할 수 있음

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/conversion-opt.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

### Conversion

TensorRT 모델로 변환할 수 있는 3가지 주요 옵션이 있다.

- using TF-TRT
- automatic ONNX conversion from `.onnx` files
- manually constructing a network using the TensorRT API (either in C++ or Python)

먼저, Tensorflow 모델을 변환할 수 있다. TensorRT integration(TF-TRT)는 모델 변환과 고수준의 런타임 API를 모두 제공하며, TensorRT가 특정 연산을 지원하지 않는 경우에는 Tensorflow 구현으로 fallback할 수 있다. 지원되는 연산은 [ONNX Operator Support](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)에서 확인할 수 있다.

두 번째로, automatic model conversion 및 deployment를 위한 보다 성능이 뛰어난 옵션은 ONNX를 사용하는 것이다. ONNX는 프레임워크에 종속되지 않으며, tensorflow 또는 pytorch 등에서 동작한다. TensorRT는 TensorRT API 또는 `trtexec`를 사용하여 ONNX 파일로부터 자동 변환을 지원한다. ONNX 변환은 `all-or-nothing` 방식이며, 모델의 모든 연산자는 TensorRT에서 지원해야 한다. 지원하지 않는 연산자는 **custom plug-ins**로 제공되어야 한다. ONNX 변환으로 TF-TRT보다 오버헤드가 적은 단일 TensorRT 엔진을 얻을 수 있다.

마지막으로, 최상의 성능과 커스터마이징을 위해 TensorRT **network definition API**를 사용하여 수동으로 TensorRT 엔진을 생성할 수도 있다. 이 방법은 변환하고자 하는 모델과 동일한 네트워크를 TensorRT 연산만 사용하여 구축하는 것이다. 프레임워크에서 학습을 수행한 뒤, 해당 모델의 파라미터(weights)만 추출하여 TensorRT 네트워크로 로드한다. 이에 대한 내용은 Developer Guide의 [Creating A Network Definition](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create_network_c)에서 다루고 있다.

### Deployment

모델을 TensorRT로 배포하는 방법에도 3가지 옵션이 있다.

- deploying within TensorFlow
- using the standalone TensorRT runtime API
- using NVIDIA Triton Inference Server

`TF-TRT`를 사용할 때, 가장 일반적인 배포 방식은 단순히 TensorFlow 내에서 배포하는 것이다. TF-TRT 변환은 TensorRT 연산이 삽입된 TensorFlow 그래프를 생성한다. 즉, python으로 다른 TensorFlow 모델과 마찬가지로 TF-TRT 모델을 실행할 수 있다.

TensorRT 런타임 API는 오버헤드가 가장 낮고, 세밀하게 추론을 제어할 수 있다. 하지만, TensorRT에서 기본적으로 제공하지 않는 연산은 플러그인(plug-in)으로 구현해야 한다 (기본적으로 제공되는 플러그인은 [link](https://github.com/NVIDIA/TensorRT/tree/main/plugin)에서 확인할 수 있다). 런타임 API로 배포하는 가장 흔한 방식은 프레임워크(tensorflow, pytorch, ...)에서 ONNX export를 사용하는 것인데, 이에 대한 내용은 아래의 [Example Deployment Using ONNX](#example-deployment-using-onnx)에서 간단하게 살펴볼 수 있다.

마지막 방법은 NVIDIA Triton Inference Server를 사용하는 것이다. Triton Inference Server는 모든 프레임워크에서 로컬 저장소 또는 Google Cloud Platform 또는 AWS S3에서 GPU/CPU 기반 인프라에 학습된 AI 모델을 배포할 수 있는 오픈소스 inference-serving 소프트웨어이다. 이에 대한 내용은 [link](https://github.com/triton-inference-server/server/blob/r22.01/README.md#documentation)에서 살펴볼 수 있다.

## Selecting the Correct Workflow

프레임워크에 따라서 몇 가지 옵션들이 있다.

1. TensorFlow 모델 및 TensorFlow 배포 - `TF-TRT`
2. TensorFlow/PyTorch 모델 - Export To `ONNX` (이 경우, C++ 또는 Python 런타임을 사용하게 됨)

<br>

# Example Deployment Using ONNX

ONNX Converion은 일반적으로 ONNX 모델을 TensorRT 엔진으로 자동으로 변환하는 가장 성능의 좋은 방법이다. 이 섹션에서는 pretrained ONNX 모델을 배포하는 맥락에서 TensorRT conversion의 5가지 기본 스텝을 살펴본다.

이 예제에서는 pretrained ResNet-50 모델을 ONNX format을 사용하여 ONNX 모델로 변환한다. ONNX는 대부분의 프레임워크로부터 추출할 수 있는 프레임워크에 구애받지 않는 model format이며, ONNX format에 대한 자세한 내용은 [link](https://github.com/onnx/onnx/blob/main/docs/IR.md)에서 확인할 수 있다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/deploy-process-onnx.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

자동으로 TensorRT 엔진으로 변환하는 방법이 두 가지 있는데, 두 방법은 필요한 모델이 서로 다르다.

- `TF-TRT` - TensorFlow 모델 필요
- `ONNX` - ONNX format으로 저장된 모델 필요

이 섹션에서는 ONNX 모델을 사용하며, 아래의 과정들을 순차적으로 보여준다.

1. Export the Model
2. Select a Batch Size
3. Select a Precision
4. Convert the Model
5. Deploy the Model

> ONNX 모델을 사용하여 변환 및 배포 방법은 링크된 주피터 노트북([link](https://github.com/NVIDIA/TensorRT/blob/f82a3220b4a3b04c2d73867ef39fee9fd57c575d/quickstart/IntroNotebooks/1.%20Introduction.ipynb))을 참조

위의 주피터 노트북에서는 batch size를 고정시켜 사용하고 있다. TensorRT는 동적으로 batch 크기를 지정할 수 있다. 다만, batch 크기가 고정되어 있으면 TensorRT가 추가적인 최적화 여지가 있다. 이렇게 동적으로 크기가 변경되는 것을 [dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)라고 부른다.

또한, precision을 FP32로 지정한다. TensorRT는 FP32 뿐만 아니라 TF32, FP16, INT8을 지원한다 ([Reduced Precision](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision) 참조).

ONNX 모델을 TensorRT 엔진으로 변환하는데 유용한 툴이 몇 가지 있는데, 가장 일반적인 툴은 ONNX 모델을 TensorRT 엔진으로 변환하고 프로파일링할 수 있는 `trtexec`이다. 이는 TensorRT 패키지에 포함되어 있으며, 아래와 같이 사용할 수 있다.

```
$ trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt
```

이렇게 변환한 TensorRT 엔진은 두 가지 타입의 TensorRT 런타임 중 하나를 선택하여 추론을 실행할 수 있다.

1. TensorRT Standalone Runtime that has C++ and Python bindings
2. A Native Integration into TensorFlow

문서에서는 standalone runtime을 호출하는 `ONNXClassifierWrapper`를 사용한다.

<br>

# TF-TRT Framework Integration

TF-TRT Integration은 TensorRT를 시작할 수 있는 간단하고 유연한 방식을 제공한다. TF-TRT는 TensorFlow 모델과 직접 동작하는 TensorRT 전용 파이썬 인터페이스이다. 이를 통해 TensorFlow로 저장된 모델을 TensorRT의 최적화된 모델(엔진)으로 변환하고 파이썬 내에서 실행할 수 있다.

TF-TRT는 변환과 다른 TensorFlow 모델과 같이 모델을 실행할 수 있는 파이썬 런타임을 모두 제공한다. 때문에 여러 가지 장점이 있는데, 특히, TF-TRT는 커스텀 플러그인을 만들지 않고도 지원되는 레이어와 지원되지 않는 레이어가 혼합된 모델을 변환할 수 있다.

링크된 노트북([link](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb))에서는 TensorFlow 2 모델을 사용하여 작업하는 기본적인 방법을 소개한다. 이 노트북에서는 미리 학습된 ResNet-50 모델과 TF-TRT를 사용하여 TensorRT 엔진으로 변환하고, TF-TRT 파이썬 런타임에서 실행한다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/tf-trt-workflow.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

<br>

# ONNX Conversion and Deployment

ONNX interchange format은 PyTorch, TensorFlow와 같은 다양한 프레임워크에서의 모델을 TensorRT 런타임으로 export할 수 있는 방법을 제공한다. ONNX를 사용하여 모델을 export하려면 모델의 연산자들이 ONNX에서 지원되어야 하며, TensorRT에서 지원하지 않는 연산자에 대해서는 플러그인 구현을 제공해야 한다.

## Exporting with ONNX

### Exporting to ONNX from TensorRT

TensorFlow 모델은 ONNX project의 `tf2onnx` 툴을 사용하여 쉽게 ONNX model로 변환할 수 있다.

링크된 노트북([link](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/3.%20Using%20Tensorflow%202%20through%20ONNX.ipynb))은 Keras/TF2 ResNet-50 모델로부터 ONNX 모델을 생성하는 방법과 생성된 ONNX 모델을 `trtexec`를 사용하여 TensorRT 엔진으로 변환하는 방법을 보여준다. 또한, TensorRT 엔진으로 데이터를 전달하여 추론하는 방법도 보여준다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/export-onnx-tf.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

### Exporting to ONNX from PyTorch

PyTorch 모델을 TensorRT로 변환하는 한 가지 방법은 PyTorch 모델을 ONNX 모델로 변한한 뒤, ONNX 모델을 TensorRT 엔진으로 변환하는 것이다. 자세한 방법은 링크된 노트북([link](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb))에서 확인할 수 있다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/export-onnx-pytorch.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

## Converting ONNX to a TensorRT Engine

ONNX 파일을 TensorRT 엔진으로 변환하는 방식에는 두 가지 방법이 있다.

1. using `trtexec`
2. using the TensorRT API

제공되는 주피터 노트북에서는 `trtexec`를 사용하고 있다. TensorRT API를 사용하는 방법은 Developer Guide에서 다룬다.

## Deploying a TensorRT Engine to the Python Runtime API

TensorRT에는 여러 런타임이 있다. 성능이 중요하다면 TensorRT API를 사용하여 ONNX 모델을 실행하는 것이 좋다. 아래 섹션에서는 C++과 Python 환경에서 TensorRT Runtime API를 사용하여 ONNX 모델을 배포하는 방법에 대해 설명한다.

<br>

# Using the TensorRT Runtime API

TensorRT API는 모델 변환과 배포에 있어서 가장 좋은 성능을 가지며 커스터마이징이 가능한 옵션 중 하나이며, C++과 Python에서 모두 제공된다.

TensorRT에는 C++과 Python binding이 포함된 standalone runtime이 있다. 일반적으로 TF-TRT Integration을 사용하여 TensorFlow에서 실행하는 것보다 성능이 더 좋고 커스터마이징이 가능하다는 장점이 있다. C++ API는 Python API보다 오버헤드가 낮지만, Python API는 Numpy 및 Scipy와 같은 파이썬 라이브러리와 함께 동작할 수 있으며 디버깅, 테스트가 더 쉽다는 장점이 있다.

문서의 튜토리얼은 TensorRT C++/Python API를 사용하여 이미지 semantic segmetation으로 설명하고 있다. 내용은 공식 문서([link](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#runtime))를 참조하길 바란다 (Quick Start Guide에 포함되기에는 TensorRT에 대해 잘 알고 있어야 이해하기 쉬울 듯 하다).

<br>

# References

- [NVIDIA TensorRT Documentation: Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
- [Basic Supported Plugins](https://github.com/NVIDIA/TensorRT/tree/main/plugin)
- [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server/blob/r22.01/README.md#documentation)
- [ONNX Intermediate Representation Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)