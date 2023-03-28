# Table of Contents

- [Table of Contents](#table-of-contents)
- [Variable Memory Space Specifiers](#variable-memory-space-specifiers)
  - [`__device__`](#__device__)
  - [`__constant__`](#__constant__)
  - [`__shared__`](#__shared__)
  - [`__grid_constant__`](#__grid_constant__)
  - [`__managed__`](#__managed__)
  - [`__restrict__`](#__restrict__)
- [References](#references)

<br>

# Variable Memory Space Specifiers

Variable memroy space specifiers는 device 변수의 메모리 위치를 나타낸다.

Device 코드 내에서 `__device__`, `__shared__`, `__constant__` 중 어느 것도 지정되지 않고 선언되는 `automatic variable`은 일반적으로 레지스터(register)에 상주하게 된다. 하지만, 특정 조건에서 컴파일러는 이를 `local memory`에 위치시키며, 이는 성능 저하를 유발시킬 수 있다.

> `Local memory`에 관한 내용은 [Maximize Memory Throughput: Local Memory](/cuda/doc/01_programming_guide/05-03_maximize_memory_throughput.md#local-memory) 참조

## `__device__`

`__device__` 지정자는 device에 상주하는 메모리를 선언한다.

아래에서 설명할 `__constant__`, `__shared__`, `__managed__`와 함께 사용될 수 있으며, 이들이 함께 지정되면 변수가 속하는 메모리 위치를 추가로 나타낸다. `__device__` 지정자만 사용된다면, 해당 변수는 다음의 특징을 갖는다.

- global memory space에 상주
- 변수가 생성된 CUDA context의 lifetime과 같음
- device마다 고유한 객체를 가짐
- grid 내 모든 스레드 또는 host로부터 액세스 가능(런타임 라이브러리 `cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`을 통해)

## `__constant__`

`__device__`와 함께 사용될 수 있으며, `__constant__`로 지정된 변수는 아래의 특징을 가진다.

- constant memory space에 상주
- 변수가 생성된 CUDA context의 lifetime과 같음
- device마다 고유한 객체를 가짐
- grid 내 모든 스레드 또는 host로부터 액세스 가능(런타임 라이브러리 `cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`을 통해)

## `__shared__`

`__device__`와 함꼐 사용될 수 있으며, 다음의 특징을 갖는다.

- 스레드 블록의 shared memory space에 상주
- 블록의 lifetime과 같음
- 블록 당 고유한 객체를 가짐
- 블록 내 모든 스레드에 의해서만 액세스 가능함
- constant address를 가지지 않음

Shared memory는 아래와 같이 external array처럼 선언할 수 있다.
```c++
extern __shared__ float shared[];
```

이 경우, shared memory array의 크기는 launch time에 execution configuration을 통해 결정된다. 이렇게 선언된 모든 변수는 동일한 메모리 주소에서 시작하기 때문에 배열 내 변수의 레이아웃은 오프셋을 통해 명시적으로 관리되어야 한다.

예를 들어, 아래와 같이 정적으로 선언된 shared memory 배열을
```c++
short array0[128];
float array1[64];
int   array2[256];
```
동일하게 동적으로 선언한 경우에서 사용하려면 다음과 같이 사용해야 한다.
```c++
extern __shared__ float array[];
__device__ void func() // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array[128];
    int*   array2 = (int*)&array1[64];
}
```

그리고, 포인터는 포인터가 가리키는 타입에 정렬되어 있어야 한다. 예를 들어, 아래 코드는 `array1`이 4바이트로 정렬되지 않기 때문에 동작하지 않는다.
```c++
extern __shared__ float array[];
__device__ void func() // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[127];
}
```

## `__grid_constant__`

`__grid_constant__`는 compute capability 7.0 이상에서 사용할 수 있으며, 커널 함수 파라미터에 사용된다. 이렇게 지정된 함수 파라미터는 아래의 특징을 갖는다.

- grid의 lifetime과 동일
- grid에 대해 private하다. 즉, host 스레드와 다른 grid의 스레드들은 액세스할 수 없다 (sub-grid 포함)
- grid 당 고유한 객체를 갖는다. 즉, 한 grid 내 모든 스레드들은 동일한 주소를 보게 된다
- read-only. `__grid_constant__` 객체나 이 객체의 sub-object를 수정하는 것은 *undefined behavior*을 유발한다 (`mutable` 멤버를 포함)

`__grid_constant__`를 사용할 때 요구사항은 다음과 같다.

- `__grid_constant__`로 지정된 커널 파라미터는 반드시 `const`-qualified non-reference type 이어야 한다
- 모든 함수 선언은 모든 `__grid_constant__` 파라미터에 대해 일치해야 한다
- 함수 템플릿 특수화(function template specialization)은 모든 `__grid_constant__` 파라미터에 대해 기본 템플릿 선언(primary template declaration)과 일치해야 한다
- 함수 템플릿 인스턴스화 지시어(directive)는 모든 `__grid_constant__` 파라미터에 대해 기본 템플릿 선언(primary template declaration)과 일치해야 한다

`__global__` 함수 파라미터의 주소를 취하면, 컴파일러는 일반적으로 thread local memory에 커널 파라미터의 복사본을 만들고 이를 사용한다. 그리고, C++ semantics를 부분적으로 지원하여 각 스레드가 함수 파라미터에 대한 자체 복사본을 수정할 수 있다. `__grid_constant__`로 지정된 `__global__` 함수 파라미터는 컴파일러가 커널 파라미터의 복사본을 thread local memory에 만들지 않도록 보장한다. 그러나 대신 파라미터 자체의 generic address를 사용한다. 따라서, local copy를 제거하여 성능을 향상시키게 된다.

```c++
__device__ void unknown_function(S const&);
__global__ void kernel(const __grid_constant__ S s) {
   s.x += threadIdx.x;  // Undefined Behavior: tried to modify read-only memory

   // Compiler will not create a per-thread thread local copy of "s":
   unknown_function(s);
}
```


## `__managed__`

`__device__`와 함꼐 사용될 수 있으며, 다음의 특징을 갖는다.

- device code와 host code 모두에서 참조될 수 있다. 예를 들어, 이 변수의 주소는 device 또는 host 함수로부터 직접 읽거나 쓸 수 있다
- 어플리케이션의 lifetime과 같다

## `__restrict__`

`nvcc`는 `__restrict__` 키워드를 통해 **restricted pointers**를 지원한다.

> Restricted pointers는 C99에서 C-type languages의 aliasing problem을 완화하기 위해 도입되었다. 이는 code re-ordering부터 common sub-expresion elimination까지 모든 종류의 최적화를 금지시킨다.

아래 예제 코드는 aliasing issue를 보여준다.
```c++
void foo(const float* a,
         const float* b,
         float* c)
{
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}
```
C-type languages에서 포인터 `a`, `b`, `c`는 앨리어싱될 수 있다. 즉, `c`를 통한 모든 write는 `a`와 `b`의 요소를 수정할 수 있다는 것을 의미한다 (`foo`를 호출할 때, `a`와 `c`에 동일한 주소를 전달하는 경우에 이에 해당한다). 이 경우에 기능적으로 정확성을 보장하기 위해 컴파일러는 `a[0]`과 `b[0]`을 레지스터에 로드하고 곱한 뒤, 결과를 `c[0]`과 `c[1]`에 저장할 수 없다. 따라서, 컴파일러는 **common sub-expression** 최적화를 적용할 수 없게 된다. 마찬가지로 컴파일러는 `c[4]`의 연산 순서를 재정렬할 수 없다. 왜냐면 앞서 계산된 `c[3]`에 의해서 `c[4]` 연산에 대한 입력이 변경될 수 있기 때문이다.

`a`, `b`, `c`를 restricted pointers로 마킹하면, 컴파일러에게 이 포인터들이 앨리어싱되지 않는다고 알려준다. 즉, `a`, `b`, `c`는 서로 동일한 주소를 가리키지 않는다는 것을 알려주는 것이다. 따라서, `c`를 통해 `a` 또는 `b`의 요소를 덮어쓰지 않는다는 것을 보장한다. 따라서, 아래와 같이 파라미터에 `__restrict__` 키워드를 추가하면, 컴파일러는 reoder 및 sub-expression eliminiation 최적화를 수행할 수 있게 된다.

```c++
void foo(const float* __restrict__ a,
         const float* __restrict__ b,
         float* __restrict__ c)
{
    float t0 = a[0];
    float t1 = b[0];
    float t2 = t0 * t1;
    float t3 = a[1];
    c[0] = t2;
    c[1] = t2;
    c[4] = t2;
    c[2] = t2 * t3;
    c[3] = t0 * t3;
    c[5] = t1;
    ...
}
```
`foo` 함수 본문은 컴파일러 최적화가 적용된 경우의 코드를 나타낸다. 코드를 보면 알겠지만, 메모리 액세스 횟수가 감소되었고, 계산의 횟수도 감소되었다.

> CUDA code에서 register pressure는 크리티컬하므로, restricted pointer를 사용하면 점유율이 줄어들어 CUDA code 성능에 부정적인 영향을 미칠 수 있다.


<br>

# References

- [NVIDIA CUDA Documentations: Variable Memory Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers)