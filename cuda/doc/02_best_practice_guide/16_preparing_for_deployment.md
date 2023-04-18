# Table of Contents

- [Table of Contents](#table-of-contents)
- [Testing for CUDA Availability](#testing-for-cuda-availability)
    - [Detecting a CUDA-Capable GPU](#detecting-a-cuda-capable-gpu)
    - [Detecting Hardware and Software Configuration](#detecting-hardware-and-software-configuration)
- [Error Handling](#error-handling)
- [Building for Maximum Compatibility](#building-for-maximum-compatibility)
- [Distributing the CUDA Runtime and Libraries](#distributing-the-cuda-runtime-and-libraries)
    - [Statically-linked CUDA Runtime](#statically-linked-cuda-runtime)
    - [Dynamically-linked CUDA Runtime](#dynamically-linked-cuda-runtime)
    - [Other CUDA Libraries](#other-cuda-libraries)
  - [CUDA Toolkit Library Redistribution](#cuda-toolkit-library-redistribution)
    - [Which Files to Redistribute](#which-files-to-redistribute)
    - [Where to Install Redistributed CUDA Libraries](#where-to-install-redistributed-cuda-libraries)
- [References](#references)

<br>

# Testing for CUDA Availability

CUDA 어플리케이션을 배포할 때, 타겟 머신이 CUDA를 사용할 수 없는 GPU이거나 설치된 NVIDIA Driver 버전이 요구하는 것보다 낮더라도 어플리케이션이 제대로 동작하도록 확인하는 것이 일반적으로 바람직하다. 물론, 고정된 구성의 단일 시스템만 고려한다면 이러한 고민을 할 필요가 없다.

### Detecting a CUDA-Capable GPU

어플리케이션이 임의의 또는 알지 못하는 머신을 타겟으로 배포된다면, 이용 가능한 GPU device가 없을 때 적절한 동작을 취하도록 CUDA-capable GPU가 존재하는지 명시적으로 테스트하는 것이 좋다. `cudaGetDeviceCount()` 함수는 가능한 GPU device의 갯수를 쿼리한다. 모든 CUDA Runtime API 함수와 마찬가지로, 이 함수는 fail할 수 있으며, 이는 정상적인 동작이다. CUDA-capable GPU가 없다면 `cudaGetDeviceCount()`는 `cudaErrorNoDevice`를 반환하며, 적절한 버전의 NVIDIA Driver가 설치되지 않았다면 `cudaErrorInsufficientDriver`를 반환한다.

시스템에서 여러 GPU를 가지고 있는 경우, GPU들의 하드웨어 버전 및 기능이 다를 수 있다. 동일한 어플리케이션에서 여러 GPU를 사용하는 경우에는 다른 세대의 하드웨어를 혼합하는 것보다 동일한 타입의 GPU를 사용하는 것이 좋다. `cudaChooseDevice()` 함수를 사용하면 원하는 기능과 가장 근접하게 일치하는 device를 선택할 수 있다.

### Detecting Hardware and Software Configuration

어플리케이션이 특정 기능을 사용하기 위해 특정 하드웨어나 소프트웨어 기능에 의존할 때, CUDA API를 통해 이용 가능한 device나 설치된 소프트웨어 버전에 대한 세부 정보를 쿼리할 수 있다.

`cudaGetDeviceProperties()` 함수는 가능한 device들의 다양한 정보를 쿼리하며, 여기에는 CUDA Compute Capability도 포함된다. CUDA 소프트웨어 API 버전은 `cudaDriverGetVersion()`과 `cudaRuntimeGetVersion()`을 통해 쿼리할 수 있다 ([Version Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION) 참조).

<br>

# Error Handling

모든 CUDA Runtime API는 `cudaError_t` 타입의 에러 코드를 반환한다. 만약 에러가 발생하지 않는다면, 반환되는 값은 `cudaSuccess`이다. 두 가지 예외가 있는데, 하나는 반환 타입이 `void`인 kernel launch이고 다른 하나는 `cudaGetErrorString()`이다. `cudaGetErrorString()`은 파라미터로 전달된 `cudaError_t` 값에 대한 설명인 문자열을 반환한다.

몇몇 CUDA API 호출과 모든 커널은 host code에 대해 비동기로 동작하기 때문에 에러 또한 비동기식으로 리포트된다. 이는 보통 host와 device가 서로 동기화할 때 발생하며, 예를 들어, `cudaMemcpy()`나 `cudaDeviceSynchronize()`를 호출하는 중에 발생할 수 있다.

모든 CUDA API 함수의 반환값을 확인하는 것이 좋으며, 에러가 발생하지 않을 것으로 예상되는 함수에 대해서도 마찬가지이다. 이렇게 하면, 어떤 에러가 발생했을 때 어플리케이션이 가능한 한 빨리 에러를 탐지하고 복구할 수 있다. `<<<...>>>` 구문을 사용하여 커널을 실행하는 동안 발생하는 에러를 확인하려면 커널을 실행하고 난 바로 다음에 `cudaGetLastError()`의 반환값을 확인해야 한다. CUDA API 에러를 확인하지 않는 어플리케이션은 GPU가 계산한 데이터가 불완전하거나 잘못되었거나 초기화되지 않는 것 등에 대해 인지하지 못한 채로 연산이 완료될 수 있다.

> CUDA Toolkit 샘플에는 다양한 CUDA API들과 함께 에러를 체크할 수 있는 헬퍼 함수들을 제공한다 (`samples/common/inc/helper_cuda.h` 참조).

# Building for Maximum Compatibility

각 세대의 CUDA-capable device는 연관된 **compute capability**가 있다. 컴파일할 때, 하나 또는 그 이상의 compute capability 버전을 `nvcc` 컴파일러에 지정할 수 있다. 이렇게 하면 어플리케이션 커널이 가능한 최상의 성능을 발휘하고 주어진 GPU 세대에서 사용 가능한 기능들을 사용할 수 있다.

어플리케이션을 여러 compute capability로 동시에 빌드하는 경우 (`nvcc`에서 여러 개의 `-gencode` 플래그 사용), 지정된 compute capability에 대한 바이너리들이 실행 파일에 결합되고, CUDA Driver는 런타임에 현재 device의 compute capability에 맞는 가장 적절한 바이너리를 선택한다. 만약 적절한 native binary (`cubin`)이 없고 PTX code가 사용 가능하다면, 커널은 해당 device에 대해 PTX에서 native `cubin`으로 **Just In Time(JIT)**로 컴파일된다. 만약 사용 가능한 PTX도 없다면 kernel launch는 실패한다.

```
/usr/local/cuda/bin/nvcc
  -gencode=arch=compute_30,code=sm_30
  -gencode=arch=compute_35,code=sm_35
  -gencode=arch=compute_50,code=sm_50
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_75,code=compute_75
  -O2 -o mykernel.o -c mykernel.cu
```

위와 같이 다양한 compute capability를 동시에 지정할 수 있다. 또는, `nvcc`의 플래그 `-arch=sm_XX`를 사용하여 `-gencode=`의 shortcut으로 사용할 수 있다. 예를 들어, `-arch=sm_70`은 `-arch=compute_70 -code=compute_70,sm_70`과 같다 (`-gencode arch=compute_70,code=\"compute_70,sm_70\"`와 동일).

하지만 `-arch=sm_XX` 옵션은 기본적으로 PTX back-end 타겟을 포함하지만(`code=compute_XX` target 때문), 한 번에 하나의 `cubin` 아키텍처만을 지정할 수 있으며 동일한 `nvcc` 커맨드에서 여러 `-arch=` 옵션을 사용할 수 없다. 위에서 `-gencode=`를 명시적으로 사용하는 이유가 바로 이 때문이다.

<br>

# Distributing the CUDA Runtime and Libraries

CUDA 어플리케이션은 device, memmory, kernel management를 처리하는 CUDA Runtime 라이브러리에 대해 빌드된다. CUDA 드라이버와 달리 CUDA 런타임은 버전 간의 forward/backward binary compatibility를 보장하지 않는다. 따라서, dynamic link를 사용하거나 CUDA Runtime에 정적으로 링크할 때, 어플리케이션과 함께 CUDA Runtime 라이브러리를 재배포하는 것이 좋다. 이렇게 하면 사용자가 어플리케이션이 빌드된 것과 동일한 CUDA Toolkit을 설치하지 않은 경우에도 실행할 수 있다.

CUDA Runtime에 정적으로 링크하면, 여러 버전이 Runtime이 동일한 어플리케이션 프로세스에 동시에 공존할 수 있다. 예를 들어, 하나의 버전의 CUDA Runtime을 사용하는 어플리케이션이 있고 그 어플리케이션의 플러그인이 다른 버전에 정적으로 링크되어 있다면, 설치된 NVIDIA Driver가 두 버전을 모두 커버할 때 이를 허용한다.

### Statically-linked CUDA Runtime

가장 쉬운 옵션이 CUDA Runtime에 정적으로 링크하는 것이다. CUDA 5.5 이상에서 `nvcc`를 사용하면 기본적으로 정적으로 링크된다. 이는 실행 파일의 용량을 좀 더 크게 만들지만, CUDA 런타임 라이브러리를 별도로 재배포할 필요없이 올바른 버전의 런타임 라이브러리 함수가 어플리케이션 바이너리에 포함되도록 한다.

### Dynamically-linked CUDA Runtime

어던 이유로 CUDA 런타임에 대한 정적 링크가 실용적이지 않다면, CUDA 런타임 라이브러리를 동적으로 링크할 수 있다.

CUDA 5.5 이상에서 `nvcc`를 사용하여 어플리케이션을 링크할 때, CUDA Runtime에 동적으로 링크하려면 링크 커맨드에 `-cudart=shared` 플래그를 추가하면 된다. 추가하지 않으면 정적 링크된 CUDA 런타임 라이브러리가 기본적으로 사용된다.

어플리케이션이 CUDA 런타임에 대해 동적으로 링크되면, 이 버전의 런타임 라이브러리는 어플리케이션과 함께 번들로 제공되어야 한다.

### Other CUDA Libraries

CUDA 런타임은 정적 링크 옵션을 제공하지만, CUDA Toolkit에 포함된 일부 라이브러리는 동적 링크 방식으로만 사용할 수 있다. CUDA 런타임 라이브러리에 동적 링크된 경우와 같이, 이러한 라이브러리는 해당 어플리케이션을 배포할 때 실행 파일과 함께 번들로 제공되어야 한다.

## CUDA Toolkit Library Redistribution

CUDA Toolkit의 EULA(End-User License Agreement)는 특정 약관에 따라 많은 CUDA 라이브러리들의 재배포를 허용한다. 이를 통해 이러한 라이브러리에 의존하는 어플리케이션이 빌드되고 테스트된 정확한 버전의 라이브러리를 재배포할 수 있으므로 다른 버전의 CUDA Toolkit이 설치된(또는 없는) 시스템의 문제를 피할 수 있다.

> NVIDIA Driver에는 적용되지 않는다. 사용자는 GPU 및 OS에 적합한 NVIDIA Driver를 직접 다운받아 설치해야 한다.

### Which Files to Redistribute

하나 이상의 CUDA 라이브러리의 동적 링크된 버전을 재배포할 때는 재배포해야 하는 파일을 정확히 식별하는 것이 중요하다. 아래는 CUDA Toolkit 5.5의 cuBLAS 라이브러리를 예시로 사용하요 보여준다.

**Linux**

Linux의 공유 라이브러리에는 라이브러리의 바이너리 호환성 레벨을 나타내는 `SONAME`이라는 문자열 필드가 있다. 어플리케이션이 빌드된 라이브러리의 `SONAME`은 어플리케이션과 함께 재배포되는 라이브러리의 파일 이름과 일치해야 한다.

예를 들어, 표준 CUDA Toolkit 설치에서 `libcublas.so` 및 `libcublas.so.5.5`는 모두 `libcublas.so.5.5.x`와 같이 cuBLAS의 특정 빌드를 가리키는 심볼릭 링크이다 (여기서 x는 빌드 번호. e.g., `libcublas.so.5.5.17`). 그러나 이 라이브러리의 `SONAME`은 `libcublas.so.5.5`로 지정된다.

```
$ objdump -p /usr/local/cuda/lib64/libcublas.so | grep SONAME
   SONAME               libcublas.so.5.5
```

이 때문에 어플리케이션을 링크할 때, `-lcublas`(버전 번호를 지정하지 않은 경우)를 사용하더라도 링크 시 발견되는 `SONAME`은 동적 로더가 어플리케이션을 로드할 때 찾을 파일의 이름이 `libcublas.so.5.5`라는 것을 나타내므로, 어플리케이션과 함께 재배포되는 파일의 이름(또는 동일한 것으로의 심볼릭 링크)이어야 한다.

`ldd` 툴은 라이브러리 검색 경로를 기준으로 동적 로더가 어플리케이션을 로드할 때 찾을 라이브러리 파일의 정확한 이름과 경로를 확인하는데 유용하다.
```
$ ldd a.out | grep libcublas
   libcublas.so.5.5 => /usr/local/cuda/lib64/libcublas.so.5.5
```

**Mac**

문서 참조. 여기서 다루진 않음.

**Windows**

Windows에서 CUDA 라이브러리의 바이너리 호환성 버전은 파일 이름의 일부로 표시된다.

예를 들어, cuBLAS 5.5에 링크된 64비트 어플리케이션은 런타임에 `cublas64_55.dll`을 찾으므로 `cublas.lib`가 어플리케이션이 링크된 파일이더라도 `cublas64_55.dll`가 해당 어플리케이션과 함께 재배포되어야 하는 파일이다. 32비트 어플리케이션인 경우, 파일은 `cublas32_55.dll`이다.

어플리케이션이 런타임 시에 찾을 것으로 예상되는 정확한 DLL 파일 이름을 확인하려면 비주얼 스튜디오의 명령 프롬프트에서 `dumpbin`을 사용하면 된다.
```
$ dumpbin /IMPORTS a.exe
Microsoft (R) COFF/PE Dumper Version 10.00.40219.01
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file a.exe

File Type: EXECUTABLE IMAGE

  Section contains the following imports:

    ...
    cublas64_55.dll
    ...
```

### Where to Install Redistributed CUDA Libraries

재배포할 라이브러리 파일이 식별되면, 어플리케이션이 찾을 수 있는 위치에 설치하도록 구성해야 한다.

Windows에서는 CUDA Runtime 또는 다른 동적 링크된 CUDA Toolkit 라이브러리가 실행 파일과 동일한 디렉토리에 있는 경우에 자동으로 이를 찾는다. Linux 및 Mac에서는 `-rpath` 링커 옵션을 사용하여 실행 파일이 시스템 경로를 검색하기 전에 이러한 라아ㅣ브러리의 로컬 경로를 검색하도록 지시해야 한다.

**Linux/Mac**

```
nvcc -I $(CUDA_HOME)/include
  -Xlinker "-rpath '$ORIGIN'" --cudart=shared
  -o myprogram myprogram.cu
```

**Windows**

```
nvcc.exe -ccbin "C:\vs2008\VC\bin"
  -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT" --cudart=shared
  -o "Release\myprogram.exe" "myprogram.cu"
```

라이브러리가 배포될 대채 경로를 지정하려면 아래와 유사한 링커 옵션을 사용하면 된다. 리눅스에서는 위와 같이 `-rpath` 옵션을 사용하면 된다. 윈도우의 경우, `/DELAY` 옵션을 사용하는데, 이를 사용하면 어플리케이션은 처음 CUDA API 함수를 호출하기 전에 `SetDllDirectory()`를 호출하여 CUDA DLLs가 포함되어 있는 경로를 지정해야 한다.

**Linux/Mac**

```
nvcc -I $(CUDA_HOME)/include
  -Xlinker "-rpath '$ORIGIN/lib'" --cudart=shared
  -o myprogram myprogram.cu
```

**Windows**

```
nvcc.exe -ccbin "C:\vs2008\VC\bin"
  -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT /DELAY" --cudart=shared
  -o "Release\myprogram.exe" "myprogram.cu"
```


<br>

# References

- [NVIDIA CUDA Documentation: Preparing for Deployment](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#preparing-for-deployment)