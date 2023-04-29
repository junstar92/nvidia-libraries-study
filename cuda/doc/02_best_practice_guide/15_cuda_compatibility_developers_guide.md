# Table of Contents

- [Table of Contents](#table-of-contents)
- [CUDA Compatibility Developer's Guide](#cuda-compatibility-developers-guide)
- [CUDA Tookit Versioning](#cuda-tookit-versioning)
- [Source Compatibility](#source-compatibility)
- [Binary Compatibility](#binary-compatibility)
  - [CUDA Binary (cubin) Compatibility](#cuda-binary-cubin-compatibility)
- [CUDA Compatibility Across Minor Releases](#cuda-compatibility-across-minor-releases)
  - [Existing CUDA Application within Minor Versions of CUDA](#existing-cuda-application-within-minor-versions-of-cuda)
- [References](#references)

<br>

# CUDA Compatibility Developer's Guide

> CUDA 호환성에 대한 내용은 아래 포스팅에서도 자세히 다루고 있음. 필요시 참조
> - [CUDA Compatibility Developer's Guide](/cuda/doc/01_programming_guide/03-03_versioning_and_compatibility.md#)

CUDA Toolkit은 새로운 기능, 성능 개선 및 중요한 버그 수정 등을 제공하기 위해 한 달 주기로 릴리즈된다. CUDA 호환성을 통해 사용자는 driver stack 전체를 업데이트하지 않고도 최신 CUDA Toolkit Software (compiler, libraries, and tools)를 업데이트할 수 있다.

CUDA software environment는 아래의 세 부분으로 구성된다.

- **CUDA Toolkit** (libraries, CUDA runtime and developer tools) - SDK for developers to build CUDA applications
- **CUDA Driver** - User-mode driver component used to run CUDA Applications (e.g., `libcuda.so` on Linux systems)
- **NVIDIA GPU device driver** - Kernel-mode driver component for NVIDIA GPUs

리눅스 시스템에서 CUDA driver와 kernel model 컴포넌트들은 NVIDIA display driver package로 함께 전달된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/CUDA-components.png" width=500px style="display: block; margin: 0 auto; background-color: white"/>

CUDA code와 non-CUDA code를 처리하기 위한 방법을 제공하는 CUDA 컴파일러(`nvcc`)는 CUDA runtime과 함께 제공되며, CUDA compiler toolchain의 일부이다. CUDA Runtime API는 개발자에서 device, kernel execution 등의 간단한 device management를 위한 high-level C++ 인터페이스를 제공하며, CUDA Driver API는 NVIDIA 하드웨어를 대상으로 하는 어플리케이션을 위한 low-level 프로그래밍 인터페이스를 제공한다.

CUDA 라이브러리들은 이러한 기술 위에 구축되어 있으며 CUDA Toolkit에 포함되어 있다. cuDNN과 같은 다른 라이브러리들은 CUDA Runtime이 필요하지만, CUDA Toolkit과 별도로 릴리즈된다.

<br>

# CUDA Tookit Versioning

CUDA 11부터 toolkit의 버전은 `.X.Y.Z` 형태의 industry-standard sementic versioning scheme를 기반으로 한다.

- `.X` (major version) - APIs have changed and binary compatibility is broken.
- `.Y` (minor version) - Introduction of new APIs, deprecation of old APIs, and source compatibility might be broken but binary compatibility is maintained.
- `.Z` (release/patch version) - New updated and patches will increment this.

Toolkit의 각 컴포넌트는 이와 같은 방식으로 버저닝되도록 권장된다. Toolkit의 각 컴포넌트들의 버전은 [table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)에서 확인할 수 있다.

CUDA 플랫폼의 호환성은 다음의 몇 가지 시나리오를 다루기 위함이다.

1. 기업 또는 데이터센터용으로 사용되는 GPU가 있는 시스템에서는 NVIDIA 드라이버 업데이트가 복잡할 수 있고, 사전에 논의가 충분히 되어야 할 수도 있다. 새로운 드라이버로의 업데이트가 늦어지면 해당 시스템의 사용자가 사용할 수 있는 새로운 기능에 액세스하지 못할 수도 있다. 새로운 CUDA 릴리즈에 대해서 드라이버 업데이트가 필요하지 않다는 것은 새로운 버전의 소프트웨어를 사용자가 더 빨리 사용할 수 있다는 것을 의미한다. (즉, 드라이버 업데이트를 하지 않고, CUDA 버전만 업데이트하여 새로운 기능을 사용할 수 있다)
2. CUDA를 기반으로 구축된 많은 소프트웨어 라이브러리 및 어플리케이션은 CUDA Runtime, 컴파일러, 또는 드라이버에 직접적으로 의존하지 않는다. 이러한 경우, 사용자 또는 개발자는 이를 사용하기 위해 전체 CUDA Toolkit 또는 드라이버를 업그레이드하지 않아도 여전히 이점을 얻을 수 있다.
3. 종속성 업그레이드는 에러가 발생하기 쉽고, 시간이 많이 걸리며, 경우에 따라 프로그램의 의미를 바꿀 수도 있다. 최신 CUDA Toolkit으로 지속적으로 컴파일한다는 것은 어플리케이션의 최종 사용자에게도 강제로 업그레이드하도록 한다는 것을 의미한다. 패키지 관리자를 통해 이 프로세스를 용이하게 할 수 있지만, 예기치 않은 문제가 발생할 수도 있고 버그가 발견되면 업그레이드 프로세스를 반복해야 한다.

CUDA는 다양한 호환성 방법을 제공한다.

1. CUDA 10에서부터 도입된 **CUDA Forward Compatible Upgrade**는 NVIDIA datacenter driver의 이전 버전이 설치된 시스템에서 새로운 CUDA 기능에 액세스하고 새로운 CUDA 릴리즈로 빌드된 어플리케이션을 실행할 수 있도록 설계되었다.
2. CUDA 11.1에서 처음 도입된 향상된 CUDA 호환성은 다음의 두 가지 이점을 제공한다.
   - CUDA Toolkit의 컴포넌트 전체에서 semantic versioning을 활용하여 하나의 CUDA minor release(예를 들어, `11.1`)용으로 어플리케이션을 빌드하고, 추후 동일한 major 제품군의 minor 릴리즈에서 동작할 수 있다 (예를 들어, `11.x`).
   - CUDA Runtime은 minimum driver version check를 완화하여 새로운 minor release 버전으로 업데이트할 때, 더이상 드라이버 업그레이드가 필요하지 않다.
3. CUDA 드라이버는 컴파일된 CUDA Application에 대해 **Backward Binary Compatibility**가 유지되도록 한다.

<br>

# Source Compatibility

**Source Compatibility**(소스 호환성)는 라이브러리에서 제공하는 일련의 보장이며, 특정 버전의 라이브러리를 사용하여 빌드된 어플리케이션은 새로운 버전의 SDK가 설치되었을 때에도 에러없이 빌드 및 실행될 수 있다는 것을 의미한다.

CUDA Driver와 CUDA Runtime은 다른 SDK 릴리즈 간에 소스 호환성을 제공하지 않는다. API는 폐기되거나 제거될 수 있다. 따라서, 이전 버전의 toolkit에서 성공적으로 컴파일된 어플리케이션이 새 버전의 toolkit에 대해 컴파일하려면 변경이 필요할 수 있다.

개발자는 deprecation과 documentation 메커니즘을 통해 현재 또는 예정된 변경점들에 대해 공지받을 수 있다. 이는 이전 toolkit을 사용하여 컴파일된 어플리케이션 바이너리가 더 이상 지원되지 않는다는 것을 의미하는 것은 아니다. 어플리케이션 바이너리는 CUDA Driver API 인터페이스에 의존하며, CUDA Driver API 자체가 toolkit 버전 간에 변경되더라도 CUDA는 CUDA Driver API 인터페이스의 binary compatibility를 보장한다.

<br>

# Binary Compatibility

라이브러리에서는 binary compatibility를 제공한다. 여기서 언급하는 라이브러리를 타겟팅하는 어플리케이션은 다른 버전의 해당 라이브러리를 동적으로 링킹될 때 동작한다.

CUDA Driver API는 버저닝된 C-style ABI가 있어서 이전 드라이버(예를 들어, CUDA 3.2)에서 실행되는 어플리케이션이 최신 드라이버(예를 들어, CUDA 11.0)에서 여전히 잘 실행되고 동작하도록 보장한다. 즉, 어플리케이션 소스가 최신 CUDA Toolkit을 사용하기 위해서 다시 컴파일되어야 하는 경우가 있더라도, 시스템에 설치된 드라이버 컴포넌트를 새로운 버전으로 교체하면 기존의 어플리케이션과 해당 기능을 항상 지원할 수 있다.

따라서, CUDA Driver API는 바이너리 호환이 가능하지만, 소스는 호환되지 않는다. 즉, OS loader는 새로운 버전을 선택하고 어플리케이션은 계속 동작하지만, 최신 SDK에 대해 어플리케이션을 다시 빌드하려면 소스 변경이 필요할 수 있다.

그리고, Minimum Driver Version의 개념에 대해 이해하는 것이 중요하다. CUDA Toolkit(and Runtime)의 각 버전은 NVIDIA driver 최소 버전을 요구한다. 해당 CUDA Toolkit 버전에서 컴파일된 어플리케이션은 오직 해당 toolkit 버전에 대한 minimum dirver version의 시스템에서만 동작한다. CUDA 11.0 이전에는 minimum driver version이 해당 버전의 CUDA Toolkit과 함께 배포된 드라이버와 동일했다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/CTK-and-min-driver-versions.png" width=600px style="display: block; margin: 0 auto; background-color: white"/>

그래서, CUDA 11.0에서 빌드된 어플리케이션인 경우에는 오직 R450 또는 이후의 드라이버가 설치된 시스템에서만 동작했다. 만약 이러한 어플리케이션을 R418 드라이버가 설치된 시스템에서 실행하면, CUDA initialization은 에러를 리턴한다 (위 이미지 참조).

예를 들어, CUDA 11.1로 컴파일된 `deviceQuery` 샘플을 R418 버전의 드라이버가 설치된 시스템에서 실행해볼 수 있다. 이 경우, CUDA initialization은 minimum driver requirement로 인해 에러를 리턴한다.
```
ubuntu@:~/samples/1_Utilities/deviceQuery
$ make
/usr/local/cuda-11.1/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o deviceQuery.o -c deviceQuery.cpp

/usr/local/cuda-11.1/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o deviceQuery deviceQuery.o

$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.165.02   Driver Version: 418.165.02   CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   42C    P0    28W /  70W |      0MiB / 15079MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+


$ samples/bin/x86_64/linux/release/deviceQuery
samples/bin/x86_64/linux/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

cudaGetDeviceCount returned 3
-> initialization error
Result = FAIL
```

<br>

## CUDA Binary (cubin) Compatibility

CUDA의 GPU 아키텍처 간의 application binary compatibility 또한 중요하다.

CUDA C++은 C++에 익숙한 유저가 프로그램을 쉽게 작성하도록 해준다. 커널은 PTX라고 부르는 CUDA Instruction Set Architecture를 사용하여 작성할 수 있다. 그러나 일반적으로는 C++과 같은 high-level 언어를 사용하는 것이 더 효과적이다. 두 경우 모두 device에서 실행하려면 커널을 `nvcc`(called cubins)를 통해 바이너리 코드로 컴파일해야 한다.

`cubins`는 아키텍처마다 다르다. `cubins`에 대한 binary compatibility는 컴파일된 버전에서 이후 minor revision의 compute capability로는 보장이 되지만, 이전 minor revision이나 major revision 간에는 보장되지 않는다. 즉, `X.y`의 compute capability에서 생성된 `cubin` object는 `X.z(z>=y)`의 compute capability에서만 실행된다.

특정 compute capability의 Device에서 코드를 실행하려면, 해당 compute capability에 호환되는 바이너리 또는 PTX 코드를 로드해야 한다. 이식성을 위해, 즉, 더 높은 compute capability의 (아직 출시되지 않은)GPU 아키텍처에 대해 코드를 실행하려고 한다면, JIT(Just-In-Time)으로 컴파일할 **PTX 코드**를 로드해야 한다.

> 어플리케이션 호환성에 대한 내용은 [Application Compatibility](/cuda/doc/01_programming_guide/03-01_compilation_with_nvcc.md#application-compatibility)에서 확인할 수 있다.

<br>

# CUDA Compatibility Across Minor Releases

CUDA 11 시작된 semantic versioning을 활용하여, CUDA Toolkit의 컴포넌트들은 toolkit의 마이너 버전에서 binary compatibility를 유지한다. 마이너 버전 간의 binary compatibility를 유지하기 위해, CUDA Runtime은 마이너 버전이 릴리즈되더라도 minimum driver version이 업그레이드되지 않는다. Minimum driver version의 업그레이드는 오직 메이저 버전 릴리즈에서만 발생한다.

새로운 toolchain에서 minimum driver version이 업그레이드되는 주요한 이유 중 하나는 PTX 코드의 JIT 컴파일과 바이너리 코드의 JIT 링크를 처리하기 위함이다.

이번 섹션에서는 CUDA 플랫폼의 호환성을 활용할 때, 몇 가지 패턴에 대한 내용들에 대해서 다룬다.

## Existing CUDA Application within Minor Versions of CUDA

```
$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   39C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

CUDA 11.1 어플리케이션(즉, cudart 11.1이 정적으로 링크됨)을 실행할 때, driver가 11.0 버전인 경우에도 성공적으로 실행하는 것을 확인할 수 있다. 즉, 시스템에서 드라이버나 기타 toolkit 컴포넌트를 업데이트할 필요가 없다. 아래 `deviceQuery` 샘플을 출력을 보면 CUDA Driver API 버전(11.0)과 Runtime API 버전(11.1)이 다른 것을 확인할 수 있다.
```
$ samples/bin/x86_64/linux/release/deviceQuery
samples/bin/x86_64/linux/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          11.0 / 11.1
  CUDA Capability Major/Minor version number:    7.5

  ...<snip>...

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.0, CUDA Runtime Version = 11.1, NumDevs = 1
Result = PASS
```

새로운 CUDA 버전을 사용하여, 사용자는 새로운 CUDA 프로그래밍 모델 API, 컴파일러 최적화 및 math library 기능의 이점을 누릴 수 있다.

하위 섹션에서는 몇 가지 주의 및 고려해야 할 사항에 대해서 다룬다.

### Handling New CUDA Features and Driver APIs

CUDA APIs 하위 집합은 새로운 드라이버가 필요없고 어떤 드라이버 종속성도 없이 호출할 수 있다. 예를 들어, `cuMemMap` APIs 또는 `cudaDeviceSynchronize()`와 같이 CUDA 11.0 이전에 도입된 모든 APIs는 드라이버 업데이트를 필요로 하지 않는다. 마이너 릴리즈에서 도입된 (새로운 드라이버가 필요한)다른 CUDA APIs를 사용하려면 fallbacks를 구현하거나 아니면 호출을 실패하는 것이 정상적이다. 유저는 릴리즈에서 도입된 새로운 CUDA APIs에 대한 헤더와 문서를 참조해야 한다.

Toolkit의 마이너 버전에 있는 기능으로 동작할 때, 어플리케이션이 이전 CUDA 드라이버에서 실행된다면 런타임에서 해당 기능을 사용하지 못할 수 있다. 이러한 기능을 이용하려는 사용자는 코드에서 동적으로 체크하여 해당 기능의 사용 가능 여부를 쿼리해야 한다. 예를 들어, 아래 코드와 같이 작성할 수 있다.
```c++
static bool hostRegisterFeatureSupported = false;
static bool hostRegisterIsDeviceAddress = false;

static error_t cuFooFunction(int *ptr)
{
    int *dptr = null;
    if (hostRegisterFeatureSupported) {
         cudaHostRegister(ptr, size, flags);
         if (hostRegisterIsDeviceAddress) {
              qptr = ptr;
         }
       else {
          cudaHostGetDevicePointer(&qptr, ptr, 0);
          }
       }
    else {
            // cudaMalloc();
            // cudaMemcpy();
       }
    gemm<<<1,1>>>(dptr);
    cudaDeviceSynchronize();
}

int main()
{
    // rest of code here
    cudaDeviceGetAttribute(
           &hostRegisterFeatureSupported,
           cudaDevAttrHostRegisterSupported,
           0);
    cudaDeviceGetAttribute(
           &hostRegisterIsDeviceAddress,
           cudaDevAttrCanUseHostPointerForRegisteredMem,
           0);
    cuFooFunction(/* malloced pointer */);
}
```

또는 어플리케이션 인터페이스를 통해 새로운 CUDA 드라이버가 없다면 전혀 동작하지 않고 에러를 리턴하도록 하는 것이 좋다.
```c++
#define MIN_VERSION 11010
cudaError_t foo()
{
    int version = 0;
    cudaGetDriverVersion(&version);
    if (version < MIN_VERSION) {
        return CUDA_ERROR_INSUFFICIENT_DRIVER;
    }
    // proceed as normal
}
```

실행 중인 드라이버에서 해당 기능이 없으면 `cudaErrorCallRequiresNewerDriver`를 리턴한다.

### Using PTX

PTX는 범용 parallel thread execution을 위한 virtual machine과 ISA를 정의한다. PTX 프로그램은 로드 시 CUDA 드라이버의 일부인 JIT 컴파일러를 통해 타겟 하드웨어 instruction set으로 번역된다. PTX는 CUDA Driver에 의해 컴파일되므로 새로운 툴체인을 사용하면 이전 CUDA 드라이버와 호환되지 않는 PTX를 생성한다. PTX가 향후 device compatibility를 위해 사용될 때는 문제가 되지 않지만, 런타임 컴파일에 사용될 때는 문제가 발생할 수 있다.

PTX를 계속 사용하는 코드의 경우에 이전 드라이버 버전에서 컴파일을 지원하려면 먼저 static ptxjitcompiler 라이브러리 또는 가상 아키텍처(e.g., `compute_80`)가 아닌 특정 아키텍처(e.g., `sm_80`)에 대한 코드 생성 옵션을 사용한 `NVRTC`를 통해 코드를 device code로 변환해야 한다. 이러한 동작을 위해서 CUDA Toolkit에는 새로운 `nvptxcompiler_static` 라이브러리가 함께 제공된다.

아래 예제 코드와 같이 사용할 수 있다.
```c++
char* compilePTXToNVElf()
{
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t elfSize, infoSize, errorSize;
    char *elf, *infoLog, *errorLog;
    int minorVer, majorVer;

    const char* compile_options[] = { "--gpu-name=sm_80",
                                      "--device-debug"
    };

    nvPTXCompilerGetVersion(&majorVer, &minorVer);
    nvPTXCompilerCreate(&compiler, (size_t)strlen(ptxCode), ptxCode);
    status = nvPTXCompilerCompile(compiler, 2, compile_options);
    if (status != NVPTXCOMPILE_SUCCESS) {
        nvPTXCompilerGetErrorLogSize(compiler, (void*)&errorSize);

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize+1);
            nvPTXCompilerGetErrorLog(compiler, (void*)errorLog);
            printf("Error log: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
    elf = (char*)malloc(elfSize);
    nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
    nvPTXCompilerGetInfoLogSize(compiler, (void*)&infoSize);

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize+1);
        nvPTXCompilerGetInfoLog(compiler, (void*)infoLog);
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    nvPTXCompilerDestroy(&compiler);
    return elf;
}
```

### Dynamic Code Generation

> [Dynamic Code Generation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#dynamic-code-generation) 참조

### Recommendations for building a minor-version compatible library

- 종속성(dependencies)를 최소화하기 위해 CUDA Runtime을 static으로 링킹하는 것이 좋다.
- 라이브러리의 soname에 대한 semantic versioning을 따라야 한다.
- 이전 드라이버와의 호환성을 유지하려면 기능을 조건부로 사용하는 것이 좋다.
- Toolkit으로부터 동적 라이브러리와 링킹할 때, 어플리케이션 링킹과 관련한 모든 컴포넌트들은 동일하거나 더 높은 버전이어야 한다. 예를 들어, CUDA 11.1 dynamic runtime에 대해 링킹하고, 11.1의 기능과 별도의 CUDA 11.2 dynamic runtime으로 링크된 별도의 shared library를 사용하는 경우, 최종 링크 단계에 CUDA 11.2 이상의 dynamic runtime이 포함되어야 한다.

### Recommendations for taking advantage of minor version compatibility in your application

특정 기능을 사용하지 못할 수 있으므로 이에 해당하는 경우에는 기능 사용 여부를 쿼리해야 한다. 이는 GPU 아키텍처, 플랫폼 및 컴파일러에 구애받지 않는 어플리케이션을 구축하는데 일반적인 것이다. 그러나 여기에는 이제 'the underlying driver'를 추가한다.

위에서 언급했듯이 어플리케이션을 빌드할 때 CUDA 런타임을 static으로 링킹하는 것이 좋다. Driver API를 직접 사용하는 경우에는 새로운 driver entry point access API (`cuGetProcAddress`)를 사용하는 것이 좋다.

Shared 또는 static 라이브러리를 사용하는 경우, 라이브러리가 minor version compatibility를 지원하는지 확인하려면 해당 라이브러리의 릴리즈 노트를 살펴보고 따르면 된다.

<br>

# References

- [NVIDIA CUDA Documentation: CUDA Compatility Developer's Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-developer-s-guide)