# Table of Contents

- [Table of Contents](#table-of-contents)
- [Difference between the driver and runtime APIs](#difference-between-the-driver-and-runtime-apis)
  - [Complexity vs. Control](#complexity-vs-control)
  - [Context Management](#context-management)
- [References](#references)

<br>

# Difference between the driver and runtime APIs

Driver APIs와 Runtime APIs는 매우 유사하며 대부분 상호교환하여 사용할 수 있다. 둘 사이에는 몇 가지 주요한 차이점이 있는데, 이는 다음과 같다.

## Complexity vs. Control

Runtime API는 device 코드를 관리하기 용이하다. 이는 Runtime API가 implicit initialization, context management, module management를 제공하기 때문이다. 하지만 코드는 더욱 단순해지지만 Driver API만큼의 세밀한 컨트롤은 불가능하다.

이에 비해, Driver API는 세분화된 제어를 제공하는데, 특히 context와 module의 로딩에 대해 세분화된 제어가 가능하다. 커널을 실행하기 위해서는 execution configuration과 커널의 파라미터를 명시적인 함수 호출을 통해 지정해야 하므로 구현하기는 훨씬 더 복잡해진다. 그러나 Runtime API는 초기화 중에 모든 커널이 자동으로 로드되고 프로그램이 실행되는 동안 로드된 상태를 그대로 유지하지만, Driver API를 사용하면 현재 로드된 모듈만 유지하거나 동적으로 모듈을 다시 로드할 수도 있다. 또한, cubin object만 처리하므로 언어와 독립적이다.

## Context Management

Driver API를 통해 컨텍스트 관리가 가능하다. 컨텍스트 관리는 Runtime API에서는 노출되지 않는다.

Runtime API에서는 스레드에 사용할 컨텍스트를 자체적으로 결정한다. Driver API를 통해 컨텍스트가 호출 스레드에 `current context`가 지정된 경우, Runtime은 해당 컨텍스트를 사용하지만, 그렇지 않은 경우에는 `primary context`를 사용한다. `Primary context`는 필요할 때마다 생성되며, 프로세스당 하나씩 생성되며 reference count되고, 더이상 참조되지 않을 때 제거된다. 한 프로세스 내에서 Runtime API의 모든 사용자는 각 스레드에 대해 컨텍스트(`current context`)가 명시적으로 지정되지 않는 한 `primary context`를 공유한다. Runtime이 사용하는 컨텍스트, 즉, `current context` 또는 `primary context`는 `cudaDeviceSynchronize()`를 사용하여 동기화하고 `cudaDeviceReset()`을 통해 제거할 수 있다.

`Primary context`를 사용하는 것(Runtime API를 사용)은 양날의 검이다. 예를 들어, 대형 소프트웨어 패키지용 플러그인을 작성하는 유저들에게 문제를 일으킬 수 있다. 모든 플러그인이 동일한 프로세스에서 실행되는 경우, 모든 플러그인이 컨텍스트를 공유하지만 서로 통신할 방법이 없을 것이다. 만약 플러그인 중 하나가 CUDA 작업을 모두 완료한 뒤, `cudaDeviceReset()`을 호출하면, 다른 플러그인들은 영문을 알지 못한채로 사용하던 컨텍스트가 제거되어 실패할 수 있다. 이 문제를 피하기 위해 CUDA 클라이언트는 Driver API를 사용하여 현재 컨텍스트를 생성하고 설정한 다음, Runtime API를 사용하여 작업할 수 있다.

다만, 컨텍스트는 device memory, extra host threads, deivce에서의 컨텍스트 스위칭 등과 같은 상당한 리소스를 소비할 수 있다. 따라서 runtime-driver context sharing은 `cuBLAS` 또는 `cuFFT`와 같이 Runtime API에서 빌드된 라이브러리와 함께 Driver API를 사용할 때 중요하다.

<br>

# References

- [NVIDIA CUDA Documentation: Difference between the driver and runtime APIs](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html#driver-vs-runtime-api)