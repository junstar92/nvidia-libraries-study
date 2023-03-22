# Table of Contents

- [Table of Contents](#table-of-contents)
- [Versioning and Compatibility](#versioning-and-compatibility)
- [CUDA Compatibility](#cuda-compatibility)
  - [Why CUDA Compatibility](#why-cuda-compatibility)
  - [Minor Version Compatibility](#minor-version-compatibility)
    - [CUDA 11 and Later Defaults to Minor Version Compatibility](#cuda-11-and-later-defaults-to-minor-version-compatibility)
    - [Application Considerations for Minor Version Compatibility](#application-considerations-for-minor-version-compatibility)
    - [Deployment Considerations for Minor Version Compatibility](#deployment-considerations-for-minor-version-compatibility)
  - [Forward Compatibility](#forward-compatibility)
    - [Forward Compatibility Support Across Major Toolkit Versions](#forward-compatibility-support-across-major-toolkit-versions)
  - [Conclusion](#conclusion)
- [References](#references)

<br>

# Versioning and Compatibility

CUDA 어플리케이션을 개발할 때 고려해야 되는 두 가지 버전이 있다.

- `Compute Capability` : Compute device(GPU)의 general specifications and features
- `Version of the CUDA driver API` : the features supported by the driver API and runtime

Driver API의 버전은 driver header file의 `CUDA_VERSION`으로 정의된다. 이를 사용하여 어플리케이션에서 현재 설치된 driver보다 더 최신의 device driver가 필요한지 확인할 수 있다. 이는 driver API가 이전 버전과 호환되기 때문에 아주 중요하다. 즉, 특정 버전의 driver API에 대해 컴파일된 어플리케이션, 플러그인, 라이브러리(CUDA runtime 포함)는 아래 그림에 나타난 대로 후속 버전의 device driver release에서 동작한다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/compatibility-of-cuda-versions.png" height=250px style="display: block; margin: 0 auto; background-color:white"/>

하지만, driver API는 이전 버전과 호환되지 않으며, 특정 버전의 driver API에서 컴파일된 어플리케이션, 플러그인, 라이브러리는 이전 버전의 device driver에서 동작하지 않는다.

지원되는 버전의 mixing 및 matching에는 제한이 있다는 점에 유의해야 한다.

1. CUDA Driver는 시스템에 하나만 설치할 수 있다. 따라서, 설치된 driver 버전은 어플리케이션, 플러그인 또는 라이브러리가 실행될 수 있는 maximum driver API 버전과 같거나 더 높은 버전이어야 한다.
2. 어플리케이션에서 사용되는 모든 플러그인과 라이브러리는 런타임에 정적으로 링크되지 않는 한 동일한 버전의 CUDA Runtime을 사용해야 한다. 이 경우에는 여러 버전의 런타임이 동일한 process space에 공존할 수 있다. `nvcc`를 사용하여 어플리케이션을 링크한다면, 기본적으로 CUDA 런타임 라이브러리의 static version이 사용되며 모든 CUDA Toolkit 라이브러리는 CUDA Runtime에 대해 정적으로 링크된다.
3. 어플리케이션에서 사용되는 모든 플러그인 및 라이브러리는 정적으로 링크되지 않는 한 런타임을 사용하는 모든 라이브러리와 동일한 버전을 사용해야 한다.

> 2번과 3번의 경우, 시스템에 설치된 런타임의 버전과 어플리케이션이 빌드된 시스템의 런타임 버전이 일치해야 한다는 것을 의미하는 것 같다.

> 버전과 관련한 호환성에 대한 내용은 아래에서 조금 더 자세히 다룬다.

<br>

# CUDA Compatibility

아래 내용은 기존 버전이 설치된 시스템에서 새로운 CUDA toolkit 컴포넌트들을 사용하는 방법에 대해서 설명한다.

> [NVIDIA CUDA Documentations: CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)에 대한 내용이다

## Why CUDA Compatibility

NVIDIA CUDA Toolkit은 개발자가 여러 시스템에서 NVIDIA GPU 가속 프로그램을 빌드할 수 있도록 한다. 여기에는 CUDA 런타임(cudart)와 다양한 CUDA 라이브러리와 tools을 포함하여 CUDA 컴파일러 툴체인이 포함되어 있다. GPU 가속 프로그램을 빌드하려면 CUDA toolkit install과 필요한 라이브러리들을 링크만 해주면 된다.

이렇게 빌드한 프로그램을 실행하려면, 시스템에는 `CUDA enabled GPU`와 빌드된 프로그램에서 사용된 CUDA Toolkit과 호환되는 `NVIDIA display driver`가 필요하다. 만약 프로그램이 해당 라이브러리에 대해 dynamic linking으로 컴파일되었다면, 시스템에는 이러한 라이브러리와 맞는 버전이 존재해야 한다.

<img src="https://docs.nvidia.com/deploy/cuda-compatibility/graphics/CUDA-components.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

위의 이미지는 CUDA 컴포넌트들을 보여준다.

모든 CUDA Toolkit에는 편의를 위해 NVIDIA display driver package도 함께 제공된다. 이 driver는 해당 버전의 CUDA Toolkit에서 제공하는 모든 기능을 지원한다. [release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)에서 CUDA Toolkit에 버전과 이를 완전히 지원하는 driver version을 확인해볼 수 있다.

> Driver package에는 user mode CUDA driver(`libcuda.so`)와 프로그램을 실행하는데 필요한 kernel mode components를 포함한다.

일반적으로 CUDA Toolkit 버전을 업그레이드하는 것은 toolkit과 driver 버전을 모두 업그레이드해야 한다.

<img src="https://docs.nvidia.com/deploy/cuda-compatibility/graphics/forward-compatibility.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

하지만 필수는 아니다. CUDA Compatibility guarantees는 특정 컴포넌트만 업그레이드할 수 있으며, 전체 시스템(toolkit + driver)을 업그레이드하지 않고 새로운 CUDA toolkit으로만 단순히 업그레이하는 방법에 대해서는 아래에서 중점적으로 다룬다.

## Minor Version Compatibility

### CUDA 11 and Later Defaults to Minor Version Compatibility

CUDA 11부터, `CUDA Toolkit release (within CUDA major release family)`로 컴파일된 프로그램은 아래 표에 표시된 **minimum required driver version**의 시스템에서 제한된 feature-set으로 실행할 수 있다. 최소한으로 요구되는 driver version은 CUDA Toolkit에 포함되어 패키지된 driver version과 다를 수 있지만, major release는 동일해야 한다(major version은 동일해야 한다는 것으로 보임).

|CUDA Toolkit|Linux x86_x64 Minimum Required Driver Version|Windows Minimum Required Driver Version|
|--|--|--|
|CUDA 12.x|>=525.60.13|>=527.41|
|CUDA 11.x|>=450.80.02|>=452.39|

> CUDA 버전에 대한 최소한의 driver version 정보는 [Release Notes: Table 2](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3)를 참조 바람

> CUDA 11.0은 위의 최소 driver version보다 이전 버전의 driver와 함께 릴리즈되었지만, 450.08.02(linux) / 452.39(windows)로 업그레이드하면 CUDA 11.x toolkit 제품군 전체에서 minor version compatibility가 가능하다.

이전 버전의 CUDA toolkit으로 빌드된 프로그램은 `binary backward compatibility`로 인해 해당 버전보다 새로운 버전의 driver에서도 동작한다. 하지만, CUDA 11 이전에는 newer CUDA toolkit으로 빌드된 프로그램은 forward compatibility package없이는 older drivers에서 지원되지 않았다 ([Forward Compatibility Support Across Major Toolkit Versions](#forward-compatibility-support-across-major-toolkit-versions) 참조).

만약, 기존에 CUDA 10.1을 사용하다가 새로운 CUDA 10.2 버전으로 업그레이드하는 경우, 요구되는 최소 driver 버전은 toolkit과 함께 패키징된 driver와 동일하다.

|CUDA Toolkit|Linux x86_x64 Minimum Required Driver Version|Windows Minimum Required Driver Version|
|--|--|--|
|CUDA 10.2|>=440.33|>=441.22|
|CUDA 10.1|>=418.39|>=418.96|

결과적으로, CUDA 11.1까지는 새로운 CUDA Toolkit이 릴리즈될 때마다 요구되는 minimum required driver version은 변경되었다. 따라서, 해당 버전들로부터 빌드된 프로그램을 지원하려면 minimum required driver version을 잘 확인해야 한다.

CUDA 11.0 Toolkit과 함께 제공된 450.80.02(linux) driver가 설치된 시스템에서는 이 driver를 계속 사용하면서 CUDA 11.1로 업그레이드가 가능하다 (CUDA 11.1에 필요한 minium required driver version도 450.80.02 이므로). 즉, `nvidia-smi`와 `deviceQuery`를 통해 아래와 같은 환경이 시스템이 가능하다는 것이다.

```
$ nvidia-smi
                
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+

...<snip>...

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

위의 출력 결과를 보면, CUDA Driver Version은 11.0이지만 Runtime Version은 11.1이라는 것을 확인할 수 있다.

Minimum required driver version에 대한 가이드는 release notes에서 확인할 수 있다. 만약 minimum required driver version이 설치되어 있지 않다면, `deviceQuery`는 아래의 에러를 출력한다.
```
$ samples/bin/x86_64/linux/release/deviceQuery
samples/bin/x86_64/linux/release/deviceQuery Starting...

CUDA Device Query (Runtime API) version (CUDART static linking)

cudaGetDeviceCount returned 3
-> initialization error
Result = FAIL
```

### Application Considerations for Minor Version Compatibility

프로그램이 실행되는 환경이 minor version 호환성에 의존할 때, 두 가지 주의 사항이 있다. 이러한 주의 사항이 제한적이라면 프로그램이 빌드된 toolkit과 동일한 minor version이나 이후 버전의 CUDA driver가 필요하다.

- **Limited feature set**
  
  CUDA Toolkit version에서 도입된 feature는 실제로 toolkit과 driver 모두에 연결되어 있을 수 있다. 이런 경우, newer version의 toolkit과 driver에 도입된 기능을 사용하는 프로그램은 이전 driver 버전에서 `cudaErrorCallRequiresNewerDriver`라는 에러를 반환할 수 있다. 이 경우, 설치된 driver도 업그레이드해야 한다.
  
  필요한 경우, 이러한 feature를 사용하는지 프로그램 내에서 명시적으로 체크하도록 하여 이러한 문제가 발생하지 않도록 할 수 있다. 이에 대한 내용은 [CUDA Compatibility Developers Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-developer-s-guide)에서 다루고 있다.
- **Applications using PTX will see runtime issues**
  
  Device code를 PTX로 컴파일하는 프로그램은 이전 driver 버전에서 동작하지 않는다. 어플리케이션에서 PTX를 사용한다면 driver 버전을 업그레이드해야 한다.
  
  [CUDA Compatibility Developers Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-developer-s-guide)와 [PTX Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility) 참조

### Deployment Considerations for Minor Version Compatibility

빌드된 프로그램은 오직 CUDA runtime에만 의존하기 때문에, 아래의 두 가지 시나리오에서 배포될 수 있다.

1. 시스템에 runtime보다 더 새로운 버전의 CUDA driver가 설치된 경우
2. 시스템에 설치된 CUDA driver보다 CUDA runtime의 버전이 더 새로운 경우 (major release는 같음)

2번 시나리오의 경우는 바로 위에서 언급한 두 가지 주의 사항에 대해서 알고 있어야 하며, 이로 인해 문제가 발생할 수 있다.

Minor version 호환성은 라이브러리의 사용 및 배포에 유연성을 제공한다. 따라서, minor version 호환성을 지원하는 라이브러리를 사용하는 프로그램은 라이브러리 버전이 다르더라도 프로그램을 다시 컴파일하지 않고 다른 버전의 toolkit 및 라이브러리가 있는 시스템에 배포할 수 있다. 이는 major release family에 속하는 모든 버전의 라이브러리에 대해 적용된다.

하지만, **라이브러리 자체에는 고려해야 할 상호 종속성이 있다**. 예를 들어, 각 `cuDNN` 버전에서는 특정 버전의 `cuBLAS`를 필요로 한다.

만약 위와 같은 이유로 minor version 호환성을 활용할 수 없다면, `Forward Compatibility` 모델을 대안으로 사용할 수 있다 (forward compatibility는 major toolkit version 간의 호환성을 위한 것이지만).

## Forward Compatibility

### Forward Compatibility Support Across Major Toolkit Versions

엄격한 테스트 및 검증으로 인해 major release version에서 NVIDIA GPU Driver 업데이트를 원치 않을 수도 있다. 이러한 시나리오를 지원하기 위해 CUDA 10.0에서 `Forward Compatibility` upgrade path를 도입했다.

<img src="https://docs.nvidia.com/deploy/cuda-compatibility/graphics/forward-compatibility-upgrade-path.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

Forward Compatibility는 NVIDIA Data Center GPU가 있는 시스템 또는 RTX 카드의 일부 NGC Server Ready SKU에만 적용된다고 한다. 일반적인 경우에는 해당되지 않으며, 주로 다른 major release families의 older NVIDIA Linux GPU driver가 설치된 시스템에서 새로운 CUDA Toolkits에서 빌드된 어플리케이션을 지원하기 위한 것이다. 여기에는 `CUDA compat package`라는 특수한 패키지를 필요로 한다.

> 일반적인 경우에는 해당하지 않으므로, 따로 자세히 다루지는 않음. 이에 대한 내용은 [Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#forward-compatibility-title)를 참조 바람.

## Conclusion

CUDA Driver는 이전 버전의 toolkit으로 빌드된 어플리케이션을 계속 지원하기 위해서 이전 버전과의 호환성을 유지한다. CUDA Toolkit 11로 빌드된 어플리케이션은 호환되는 minor driver version을 사용하여 해당 major release 내의 모든 driver에서 지원된다.

일반적인 경우는 아니지만, CUDA Forward Compatibility package를 통해 minimu required driver 버전을 만족하지 않는 older driver가 시스템에 설치된 경우에도 newer toolkit으로 빌드된 어플리케이션을 실행할 수 있다.

또한, 전체 CUDA Toolkit 또는 driver를 업그레이드하지 않고, CUDA 라이브러리를 빠르게 업그레이드할 수 있다. 이는 이러한 라이브러리들이 CUDA runtime, 컴파일러, driver에 직접적인 종속성이 없기 때문이다.

<br>

# References

- [NVIDIA CUDA Documentations: Versioning and Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#versioning-and-compatibility)
- [NVIDIA CUDA Documentations: CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3) (CUDA Minor Version Compatibility Table)