# Table of Contents

- [Table of Contents](#table-of-contents)
- [Read-Only Data Cache Load Function](#read-only-data-cache-load-function)
- [Load Functions Using Cache Hints](#load-functions-using-cache-hints)
- [Store Functions Using Cache Hints](#store-functions-using-cache-hints)
- [References](#references)

<br>

# Read-Only Data Cache Load Function

Compute capability 5.0 이상의 device부터는 read-only data cache load function `__ldg()`을 지원한다.

```c++
T __ldg(const T* address);
```

위 함수는 `address` 위치의 `T` 타입의 값을 반환하며, `T`로는 기본 타입들과 built-in vector 타입 그리고 `cuda_fp16.h`와 `cuda_bf16.h`에서 제공하는 `half` 타입들이 가능하다.

Global memory에서 read-only 데이터에 대해서 `__ldg()`를 사용하면 unified L1/texture 캐시에 캐싱하여 읽을 수 있다. 이때, 컴파일러가 read-only condition이 충족되었다고 감지하면 `__ldg()`를 통해 이를 읽는다. 하지만, 일부 데이터에 대해서는 read-only condition이 충족되는지 감지를 못할 수도 있다. `const`와 `__restrict__` 한정자를 함께 사용하면 이러한 데이터를 로드하는데 컴파일러가 read-only condition을 감지할 가능성을 높인다.

<br>

# Load Functions Using Cache Hints

아래의 load function들은 compute capability 5.0 이상의 device에서 지원한다.
```c++
T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);
```

각 함수에 해당하는 cache operator은 [PTX ISA: Table 27](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id48)에서 설명하고 있다. 간단히 요약하면 다음과 같다.

- `__ldcg()` : cache at global level (cache in L2 and below, not L1) (`.cg`)
- `__ldca()` : cache at all levels, likely to be accessed again (`.ca`)
- `__ldcs()` : cache streaming, likely to be accessed once (`.cs`)
- `__ldlu()` : Last use (`.lu`)
- `__ldcv()` : Don't cache and fetch again (consider cached system memory lines stale, fetch again) (`.cv`)

<br>

# Store Functions Using Cache Hints

아래 store function들 또한 compute capability 5.0 이상의 device에서 지원한다.
```c++
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```

마찬가지로 각 함수에 대응하는 cache operator를 [PTX ISA: Table 28](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id49)에서 설명하고 있다.

<br>

# References

- [NVIDIA CUDA Documentations: Read-Only Data Cache Load Function](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function)
- [NVIDIA CUDA Documentations: Load Functions Using Cache Hints](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#load-functions-using-cache-hints)
- [NVIDIA CUDA Documentations: Store Functions Using Cache Hints](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#store-functions-using-cache-hints)
- [NVIDIA CUDA Documentations: PTX ISA - Cache Operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)