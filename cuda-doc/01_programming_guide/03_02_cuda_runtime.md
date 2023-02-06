# Table of Contents

- [Table of Contents](#table-of-contents)
- [CUDA Runtime](#cuda-runtime)
- [Initialization](#initialization)
- [References](#references)

<br>

# CUDA Runtime

CUDA C++은 C++ 언어에 익숙한 사람들이 쉽게 프로그램이 작성될 수 있도록 편리한 방법을 제공한다. CUDA C++은 C++ 언어에서 몇 가지 확장된 것들과 런타임 라이브러리로 구성되어 있다.

핵심적인 내용들은 [Programming Model](02_programming_model.md)에서 소개했으므로, 이를 참조하면 된다. 여기서는 C++ 함수와 같은 커널 함수를 정의하고 몇 가지 새로운 문법을 사용하여 그리드와 블록 차원을 설정하여 이 함수를 호출하는 방법을 소개했다.

이 챕터부터 이어서 [**CUDA Runtime**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime)에 대해서 살펴볼텐데, runtime에서는 host에서 실행되는 몇 가지 C/C++ 기능(device memory 할당/해제, host/device 간 메모리 전송 등)을 제공한다. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)에 제공되는 기능들과 특징들을 살펴볼 수 있다.

Runtime은 더 낮은 계층의 C API인 [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) 위에 구성된다. CUDA driver API 또한 프로그램 내에서 사용할 수 있다. CUDA Driver API는 CUDA 컨텍스트와 같은 더 낮은 레벨의 컨셉들을 위한 부가적인 제어를 제공한다. 대부분의 경우 Driver API를 사용하지 않으며, Runtime API를 주로 사용한다. Runtime API를 사용하면 컨텍스트나 모듈의 관리가 내부적으로 제어되므로, 간결한 코드를 작성할 수 있다. 물론, Runtime은 driver API와 상호운용(interoperable)이 가능하지만, 정말 필요한 경우에만 Driver API를 사용한다.

CUDA Runtime은 `cudart` 라이브러리에서 구현되며, 이 라이브러리는 `cudart.lib` 또는 `libcudart.a`를 통해 정적 링크되거나 `cudart.dll` 또는 `libcudart.so`를 통해 동적 링크된다.

<br>

# Initialization

CUDA 런타임에 명시적인 초기화 함수는 없으며, 런타임 함수가 처음 호출될 때 초기화된다. 구체적으로 말하면, 에러 핸들링과 버전 관리 함수들 이외의 함수들을 처음 호출할 때 초기화된다. 런타임 함수 호출 시간을 측정할 때와 런타임에 대한 첫 번째 호출의 에러 코드를 해석할 때, 이를 염두해야 한다.

CUDA 런타임은 시스템에서 각 GPU 장치에 대한 CUDA [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)를 생성한다. 이 컨텍스트는 장치의 기본 컨텍스트(primary context)이며, 앞서 언급했듯이 첫 번째 런타임 함수에서 초기화된다.

컨텍스트는 프로그램 내의 모든 host threads 간에 공유된다. 컨텍스트 생성의 일부가 만약 필요하다면, device code는 just-in-time(JIT) 컴파일되며 device memory에 로드된다. 모든 과정은 숨겨져 있지 않으며, Driver API를 통해 GPU 장치의 기본 컨텍스트에 액세스할 수 있다.

Runtime API 중 하나인 `cudaDeviceReset()`을 호출하면 host thread가 현재 동작 중인 GPU device의 기본 컨텍스트를 파괴한다. 그리고, 이 GPU device를 가지고 있는 host thread에서(어떤 스레드라도 가능한 것으로 보인다) 이 다음에 호출되는 런타임 함수로부터 해당 GPU device의 새로운 기본 컨텍스트를 생성한다.

> CUDA 인터페이스는 host 프로그램의 초기화 단계에서 초기화되고, 프로그램이 종료될 때 파괴되는 global state를 사용한다. CUDA Runtime은 이 상태에 유효한지 체크할 수 없기 때문에, 프로그램이 초기화되거나 종료되는 중에 CUDA 인터페이스를 사용하면 undefined behavior이 발생한다.
>
> CUDA 12.0부터는 `cudaSetDevice(index)` 호출을 통해 현재 host thread에서 사용할 GPU device를 설정하고 runtime을 명시적으로 초기화한다. CUDA 12.0 이전 버전에서 `cudaSetDevice(index)`는 단순히 index에 해당하는 GPU 장치를 설정할 뿐, 초기화는 첫 번째 런타임 호출에서 발생했다. 따라서, `cudaSetDevice(index)`를 호출할 때, 리턴 값을 체크하여 초기화 에러를 확인하는 것이 중요하다.

<br>


# References

- [NVIDIA CUDA Documentations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization)