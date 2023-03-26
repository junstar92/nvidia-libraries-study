# Table of Contents

- [Table of Contents](#table-of-contents)
- [Function Execution Space Specifiers](#function-execution-space-specifiers)
  - [`__global__`](#__global__)
  - [`__device__`](#__device__)
  - [`__host__`](#__host__)
  - [Undefined behavior](#undefined-behavior)
  - [`__noinline__` and `__forceinline__`](#__noinline__-and-__forceinline__)
- [References](#references)

<br>

# Function Execution Space Specifiers

`Function execution space specifiers`는 함수가 host 또는 device에서 수행되는지와 host 또는 device로부터 호출 가능한 지를 나타낸다.

## `__global__`

`__global__`는 커널 함수를 선언한다. 이 함수는

- device에서 실행되고
- host에서 호출할 수 있으며
- compute capability 5.0 이상의 device에서는 device에서도 호출할 수 있다 ([CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism) 참조)

`__global__` 함수는 반드시 void return type이어야 하며, 클래스의 멤버가 될 수 없다.

`__global__` 함수 호출에는 반드시 execution configuration을 지정해야 한다.

`__global__` 함수 호출은 비동기식이다. 즉, device에서 이 함수의 수행이 끝나기 전에 제어권을 host로 반환한다.

## `__device__`

`__device__`로 지정한 함수는 아래의 특징을 갖는다.

- device에서 실행되고
- 오직 device에서 호출할 수 있다

`__global__`과 `__device__` 지정자는 함께 사용될 수 없다.

## `__host__`

`__host__`로 지정한 함수는 아래의 특징을 갖는다.

- host에서 실행되고
- 오직 host에서만 호출할 수 있다

`__host__`로만 지정된 함수를 선언하는 것과 `__host__`, `__device__`, `__global__` 중 어느 것도 지정하지 않고 선언하는 것은 서로 동일하다. 이 경우, 이렇게 선언된 함수는 오직 host에서만 컴파일된다.

`__global__`과 `__host__`는 함께 사용될 수 없다.

`__device__`와 `__host__`는 함께 사용될 수 있다. 하지만, 이 경우에는 host와 device에 모두에 대해 컴파일된다. `__CUDA_ARCH__` 매크로를 아래 예제 코드와 같이 사용하면 host와 device 간, 또는 compute capability가 다른 device간의 코드를 분기시킬 수 있다.

```c++
__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 800
    // device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700
    // device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
    // device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
    // device code path for compute capability 5.x
#elif !defined(__CUDA_ARCH__)
    // host code path
#endif
}
```

## Undefined behavior

'cross-execution space' 호출은 몇몇 조건에서 undefined behavior이 있다고 한다. 해당 조건에 대한 의미가 명확하지 이해되지 않아 따로 여기서 정리는 하지 않으며, [문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#undefined-behavior)를 참조하길 바란다.

> 해당 내용에 대한 개인적인 추측(또는 생각)은 다음과 같다.
> - `__global__`, `__device__`, or `__host__ __device__` 함수에서 `__host__` 함수 호출
> - `__host__` 함수에서 `__device__` 함수 호출
>
> 위와 같은 조건의 호출에서 undefined behavior을 가진다고 언급하고 있다. host와 device 중 어디에서 컴파일되는가에 따라 `__CUDA_ARCH__`의 정의 유무가 달라질텐데, 아마도 정의 유무 각각에 따른 위 조건에서 undefined behavior을 갖는다는 것을 의미하는 것으로 추측된다.

## `__noinline__` and `__forceinline__`

컴파일러는 적절하다고 판단될 때, 모든 `__device__` 함수를 인라인한다.

`__noinline__` function qualifier은 가능한 경우 함수를 인라인하지 않도록 컴파일러에게 힌트를 주는 정도로 사용할 수 있다.

`__forceinline__` function qualifie은 컴파일러가 함수를 인라인하도록 강제하는데 사용할 수 있다.

`__noinline__`과 `__forceinline__`은 함께 사용될 수 있으며, 두 한정자 모두 인라인 함수에는 적용할 수 없다.

<br>

# References

- [NVIDIA CUDA Documentations: Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers)