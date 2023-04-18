# Table of Contents

- [Table of Contents](#table-of-contents)
- [Understanding the Programming Environment](#understanding-the-programming-environment)
- [CUDA Compute Capability](#cuda-compute-capability)
- [Additional Hardware Data](#additional-hardware-data)
- [Which Compute Capability Target](#which-compute-capability-target)
- [CUDA Runtime](#cuda-runtime)
- [References](#references)

<br>

# Understanding the Programming Environment

각 세대의 NVIDIA 프로세서에서는 CUDA가 활용할 수 있는 새로운 기능에 GPU에 추가된다. 따라서, 각 아키텍처의 특성을 이해하는 것이 중요하다.

프로그래머는 두 개의 버전을 알고 있어야 하는데, 하나는 [compute capability](#cuda-compute-capability)이고, 다른 하나는 CUDA Runtime과 CUDA Driver APIs의 버전이다.

<br>

# CUDA Compute Capability

**Compute capability**은 하드웨어의 기능을 설명하고, 기타 사양(e.g., maximum number of threads per block 등)뿐만 아니라 장치에서 지원하는 instruction set을 반영한다. 더 높은 compute capability 버전은 더 낮은(이전) 버전의 상위 집합이므로 이전 버전과 호환된다.

Device에 있는 GPU의 compute capability는 CUDA 샘플의 `deviceQuery`와 같이 프로그래밍 방식으로 쿼리할 수 있다. 출력은 다음과 같다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/sample-cuda-configuration-data.png" width=900px style="display: block; margin: 0 auto; background-color: white"/>

위 정보는 `cudaGetDeviceProperties()`를 호출하여 얻을 수 있으며, 전달한 구조체(`cudaDeviceProp`)를 통해 정보를 얻을 수 있다.

위 출력에서 확인할 수 있듯이 compute capability는 major와 minor의 두 가지 버전이 있다. 위 출력을 통해 device 0이 compute capability 7.0이라는 것을 알 수 있다.

각 Compute capability에 대한 정보는 아래 문서에서 자세히 확인할 수 있다.

- [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

<br>

# Additional Hardware Data

특정 하드웨어 기능은 compute capability로 구분할 수 없다. 예를 들어, host와 device 간의 asynchronous data trasfers와 kernel execution의 오버랩 기능은 대부분의 GPU에서 사용할 수 있지만, compute capability와 관계없이 모든 GPU에서 사용할 수 있는 것은 아니다. 이러한 경우에는 `cudaGetDeviceProperties()`를 호출하여 장치가 특정 기능을 사용할 수 있는지 여부를 확인해야 한다. 예를 들어, 이 함수 호출로 반환되는 구조체의 `asyncEngineCount` 멤버는 kernel execution과 data transfers가 가능한지 여부를 나타낸다. 마찬가지로 `cudaMapHostMemory` 멤버는 zero-copy data transfers를 수행할 수 있는지 여부를 나타낸다.

<br>

# Which Compute Capability Target

특정 버전의 NVIDIA 하드웨어 및 CUDA 소프트웨어를 타겟으로 하려면 `nvcc`의 `-arch`, `-code`, `-gencode` 옵션을 사용하면 된다. 예를 들어, warp shuffle instruction을 사용하는 코드는 `-arch=sm_30`(또는 더 높은 compute capability)로 컴파일해야 한다.

> 여러 세대의 CUDA-capable device를 동시에 지원하도록 코드를 빌드하기 위한 빌드 플래그는 [Building for Maximum Compatibility](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility)를 참조

<br>

# CUDA Runtime

CUDA 소프트웨어 환경의 host runtime 컴포넌트는 host 함수에 의해서만 사용할 수 있으며, 아래에 대한 내용들을 처리하는 함수들을 제공한다.

- Device management
- Context management
- Memory management
- Code module management
- Execution control
- Texture reference management
- Interoperability with OpenGL and Direct3D

Low-level의 CUDA Driver API와 비교했을 때, CUDA Runtime은 암시적인 initialization, context management, device module management를 제공하여 device management를 용이하게 한다. `nvcc`에 의해 생성된 C++ host code는 CUDA Runtime을 활용하므로 이 코드와 링크되는 어플리케이션은 CUDA Runtime에 의존한다. 비슷하게, cuBLAS, cuFFT, 또는 다른 CUDA Toolkit 라이브러리를 사용하는 모든 코드는 라이브러리 내부에서 사용하는 CUDA Runtime에 의존한다.

CUDA Runtime은 커널이 시작되기 전에 kernel loading과 kernel parameter 및 launch configuration을 처리한다. 암시적인 driver version checking, code initialization, CUDA context management, CUDA module management (cubin to function mapping), kernel configuration, parameter padding은 모두 CUDA Runtime에 의해 수행된다.

CUDA Runtime은 아래의 두개의 파트로 구성된다.

- A C-style function interface (`cuda_runtime_api.h`)
- C++-style convenience wrappers (`cuda_runtime.h`) built on top of the C-style functions

> CUDA Runtime에 대한 내용은 아래 포스팅에서도 다루고 있다.
> - [CUDA Runtime](/cuda/doc/01_programming_guide/03-02_cuda_runtime.md)

<br>

# References

- [NVIDIA CUDA Documentation: Understanding the Programming Environment](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#understanding-the-programming-environment)