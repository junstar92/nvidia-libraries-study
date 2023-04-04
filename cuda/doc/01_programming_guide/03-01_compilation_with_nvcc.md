# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [Compilation Workflow](#compilation-workflow)
  - [Offline Compliation](#offline-compliation)
  - [Just-in-Time Compilation](#just-in-time-compilation)
- [Binary Compatibility](#binary-compatibility)
- [PTX Compatibility](#ptx-compatibility)
- [Application Compatibility](#application-compatibility)
- [C++ Compatibility](#c-compatibility)
- [64-Bit Compatibility](#64-bit-compatibility)
- [References](#references)

# Intro

커널(kernel) 함수는 PTX라는 CUDA instruction set architecture를 사용하여 작성될 수 있다. 하지만 보통 C++과 같은 high-level 프로그래밍 언어를 사용하는 것이 더 효과적이라고 한다. 아무튼 PTX 또는 C++로 작성된 커널은 `nvcc`라는 툴을 사용하여 바이너리 코드로 컴파일하고 디바이스에서 실행할 수 있다.

`nvcc`는 C++ 또는 PTX 코드 컴파일 프로세스를 간단히 해주는 compiler driver이다. 간단하고 익숙한 커맨드라인 옵션을 제공하며, 다양한 컴파일 단계를 구현하는 툴들을 호출하여 실행한다. 이번 포스팅에서는 `nvcc`의 workflow와 커맨드 옵션에 대한 개요를 살펴본다.

# Compilation Workflow

## Offline Compliation

`nvcc`로 컴파일되는 소스 파일은 host code와 device code가 혼재되어 있다. `nvcc`의 기본 workflow는 host code로부터 device code를 분리시킨 다음, 다음을 수행한다.

- device code를 어셈블리 형태(PTX code)와/또는 바이너리 형태(`cubin` object)로 컴파일한다.
- 그리고, host code에서 앞서 컴파일된 커널을 실행하는 `<<<...>>>` 문법(execution configuration)을 필요한 CUDA 런타임 함수 호출로 대체하도록 수정한다.

이렇게 수정된 host code는 다른 tool을 사용하여 컴파일해야 할 c++ code이거나 마지막 컴파일 단계 동안 `nvcc`가 host compiler를 호출하여 생성된 object code이다.

어플리케이션은 

- 컴파일된 host code에 링크할 수 있고(가장 일반적인 경우),
- 또는, 수정된 host code를 무시하고 PTX code 또는 `cubin` object를 읽고 실행하는 CUDA driver API를 사용할 수도 있다.

아래 그림은 컴파일 프로세스 과정을 보여준다 ([link](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#the-cuda-compilation-trajectory)).

<img src="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/_images/cuda-compilation-from-cu-to-executable.png" width=700px style="display: block; margin: 0 auto; background-color:white"/>

## Just-in-Time Compilation

런타임에서 어플리케이션에 의해 로드되는 모든 PTX code는 device driver에 의해서 바이너리 코드로 추가로 컴파일된다. 이를 *just-in-time compilation* 이라고 부른다. 비록 어플리케이션의 로드 타임은 증가하지만, 새로운 device driver에서 제공되는 새로운 컴파일러 향상으로부터 이점을 얻을 수 있다. 또한, 컴파일된 시점에서 존재하지 않았던 device에서 어플리케이션을 실행할 수 있는 유일한 방법이기도 하다.

> Device driver가 어플리케이션의 PTX code를 JIT(just-in-time) 컴파일할 때, 반복적인 컴파일을 피하기 위해 생성된 바이너리 코드 복사본을 캐싱한다. *Compute cache*라고 불리는 이 캐시는 device driver가 업그레이드될 때 무효가 되어, device driver에 내장된 새로운 JIT 컴파일러 개선을 적용할 수 있다.

> 환경 변수를 통해 JIT 컴파일을 제어할 수 있으며, 관련 환경 변수는 [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)에서 확인할 수 있다.

> `nvcc`를 사용하여 CUDA C++ device code를 컴파일하는 대신, `NVRTC`를 사용하여 런타임에 CUDA C++ device code를 PTX로 컴파일할 수 있다. `NVRTC`는 CUDA C++용 런타임 컴파일 라이브러리이다. 이에 대한 내용은 [NVRTC User guide](https://docs.nvidia.com/cuda/nvrtc/index.html)에서 확인할 수 있다.

<br>

# Binary Compatibility

바이너리 코드는 **architecture-specific** 이다. 즉, `cubin` object 파일은 `-code` 컴파일 옵션을 통해 타겟 아키텍처를 지정하여 생성된다. 예를 들어, `-code=sm_80`을 포함한 컴파일은 compute capability 8.0 device용 바이너리 코드를 생성한다. 바이너리 호환성은 마이너 버전과 그 이후 마이너 버전으로는 호환을 보장하지만, 이전 마이너 버전 또는 다른 메이저 버전 간의 호환은 보장하지 않는다. 즉, compute capability `X.y`에서 생성된 cubin object는 오직 compute capability `X.z`($z\geq y$ 인 경우)에서만 실행된다.

> 바이너리 호환성은 desktop에서만 지원되며, Tegra에서는 지원되지 않는다. 또한, desktop과 Tegra 간의 바이너리 호환성도 지원되지 않는다.

<br>

# PTX Compatibility

몇몇 PTX instruction은 더 높은 compute capability의 device에서만 지원된다. 예를 들어, [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)는 오직 compute capability 5.0 이상 device에서만 지원된다. `-arch` 컴파일러 옵션은 C++ 코드를 PTX 코드로 컴파일할 때 가정되는 compute capability를 지정한다. 예를 들어, warp shuffle을 포함하는 코드는 반드시 `-arch=compute_30`(?)(or higher)로 컴파일되어야 한다.

> `-arch=compute_30`이 아니라 `-arch=compute_50`일 것 같다.. 아마 문서상의 오류로 추정된다.

몇몇 지정된 compute capability용으로 생성된 PTX 코드는 항상 같거나 더 높은 compute capability의 바이너리 코드로 컴파일될 수 있다. 이전 PTX 버전으로부터 컴파일된 바이너리는 일부 하드웨어 기능을 사용하지 않을 수도 있다. 예를 들어, compute capability 6.0(Pascal)용으로 생성된 PTX로부터 compute capability 7.0(Volta) device을 타겟으로 컴파일되는 바이너리는 Tensor Core instruction을 사용하지 않는다 (Pascal에서는 사용할 수 없었기 때문). 결과적으로 최신 버전의 PTX로 바이너리를 생성한 것보다 성능이 더 떨어질 수 있다.

<br>

# Application Compatibility

특정 compute capability의 device에서 실행하기 위해서, 어플리케이션은 해당 compute capability와 호환되는 바이너리 또는 PTX 코드를 로드해야 한다. 특히, 이후에 출시되는 더 높은 compute capability의 아키텍처의 경우에는 미리 바이너리 코드를 생성할 수 없기 때문에 just-in-time으로 컴파일되는 PTX code를 로드해야 한다.

CUDA C++ 어플리케이션에 포함되는 PTX와 바이너리 코드는 `-arch`와 `-code` 컴파일 옵션 또는 `-gencode` 컴파일 옵션에 의해 제어된다.

```
nvcc x.cu
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_70,code=\"compute_70,sm_70\"
```
예를 들어, 위의 명령어는 compute capability 5.0과 6.0에 호환되는 바이너리 코드를 포함하고(첫 번째, 두 번째 `-gencode` 옵션), compute capability 7.0에 호환되는 PTX와 바이너리 코드를 포함(세 번째 `-gencode` 옵션)한다.

Host code는 런타임 시에 로드 및 실행하기에 가장 적절한 코드를 자동으로 선택하기 위해 생성된다. 위에서 살펴본 커맨드의 경우에는 다음의 코드들이 선택될 수 있다.

- 5.0 binary code for devices with compute capability 5.0 and 5.2,
- 6.0 binary code for devices with compute capability 6.0 and 6.1,
- 7.0 binary code for devices with comptue capability 7.0 and 7.5,
- PTX code which is compiled to binary code at runtime for devices with compute capability 8.0 and 8.6

예를 들어, `x.cu` 코드는 compute capability 8.0 이상의 device에서만 지원되는 warp reduction 연산을 사용하는 최적화된 코드 경로를 가질 수 있다. 이 경우, `__CUDA_ARCH__` 매크로를 사용하여 compute capability에 맞는 다양한 코드 경로를 구별할 수 있다 (`__CUDA_ARCH__`는 device code에서만 정의된다). 예를 들어, `-arch=compute_80`으로 컴파일될 때, `__CUDA_ARCH__`는 `800`이다.

> Driver API를 사용하는 어플리케이션은 코드를 분리된 파일에 컴파일하고, 런타임 시에 가장 적절한 파일을 명시적으로 로드 및 실행해야 한다.

Volta 아키텍처에서는 [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling-7-x)을 도입했다. 따라서, 이전 아키텍처에서의 SIMT 스케줄링 동작을 기반으로 구현된 코드의 경우, Independent Thread Scheduling으로 인해 잘못된 결과를 초래할 수 있다. 이러한 마이그레이션을 지원하기 위해서 `-arch=compute_60 -code=sum_70`을 사용하여 Volta에 호환되는 바이너리 코드를 생성하면서 Pascal의 스레드 스케줄링을 선택하도록 할 수 있다.

> `nvcc`에서는 `-arch`, `-code`, `-gencode` 컴파일 옵션에 대한 shortcut을 제공한다. 예를 들어, `-arch=sm_70`은 `-arch=compute_70 -code=compute_70,sm_70`과 같다 (`-gencode arch=compute_70,code=\"compute_70,sm_70\"`와 동일).

<br>

# C++ Compatibility

컴파일러의 프론트엔드에서는 C++ syntax 규칙에 따라 CUDA 소스 파일을 처리한다. Full C++의 경우에는 host code에 대해서 지원하며, device code에서는 일부 C++ 기능만 완전히 지원된다.

> 지원되는 C++ 기능은 [C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support)에서 확인할 수 있다.

<br>

# 64-Bit Compatibility

64-bit 버전의 `nvcc`는 device code를 64-bit 모드로 컴파일한다 (즉, 포인터가 64-bit). 64-bit mode로 컴파일된 device code는 오직 64-bit mode로 컴파일된 host code와 함께 지원된다.

<br>

# References

- [NVIDIA CUDA Documentations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc)
- [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)