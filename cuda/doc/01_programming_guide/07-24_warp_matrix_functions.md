# Table of Contents

- [Table of Contents](#table-of-contents)
- [Warp Matrix Functions](#warp-matrix-functions)
- [Description](#description)
  - [`fragment`](#fragment)
  - [`load_matrix_sync`](#load_matrix_sync)
  - [`store_matrix_sync`](#store_matrix_sync)
  - [`fill_fragment`](#fill_fragment)
  - [`mma_sync`](#mma_sync)
- [Alternate Floating Point](#alternate-floating-point)
- [Double Precision](#double-precision)
- [Sub-byte Operations](#sub-byte-operations)
  - [`bmma_sync`](#bmma_sync)
- [Restrictions](#restrictions)
- [Element Types and Matrix Sizes](#element-types-and-matrix-sizes)
- [Example](#example)
- [References](#references)

<br>

# Warp Matrix Functions

CUDA C++ warp matrix operation으로 텐서 코어(Tensor Cores)를 활용하여 `D=A*B+C` 형태의 행렬 문제를 가속할 수 있다. 이러한 연산은 compute capability 7.0 이상의 device에서 mixed-precision floating point 데이터에 대해 지원된다. 이 연산은 조건문에서도 허용되는데, warp 내 모든 스레드의 협력을 필요로 한다. 또한, warp 전체 내에서 동일하게 평가되는 조건인 경우일 때만 허용되며 그렇지 않으면 프로그램은 멈추게 된다.

> `wmma`는 Warp Matrix Multiply-Accumulate를 의미한다.

# Description

WMMA와 관련된 모든 함수와 타입들은 `nvcuda::wmma` 네임스페이스에 정의되어 있다.

> sub-byte operation은 preview이다. 따라서, 이와 관련된 데이터 구조 및 API는 변경될 수 있다. 이 기능들은 `nvcuda::wmma::experimental` 네임스페이스에 정의되어 있다.

```c++
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

## `fragment`

```c++
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
```

`wmma::fragment`는 warp 내 모든 스레드에 분산된 행렬 섹션을 포함하는 오버르도된 클래스이다. `fragment` 내부 storage로의 행렬 요소 매핑은 지정되어 있지 않으며 이후 아키텍처에서 변경될 수 있다.

`fragment`의 템플릿 인자는 특정 조합만 허용된다. 첫 번째 템플릿 파라미터는 fragment가 행렬 연산에 참여할 방법을 지정한다. `Use`에 허용되는 값은 아래와 같다.

- `matrix_a` - fragment가 첫 번째 multiplicand, `A`로 사용될 때
- `matrix_b` - fragment가 두 번째 multiplicand, `B`로 사용될 때
- `accumulator` - fragment가 source accumulator(`C`) 또는 destination accumulator(`D`)로 사용될 때

`m`, `n`, `k`의 크기는 multiply-accumulate 연산에 참여하는 warp-wide matrix tiles의 모양을 나타낸다. 각 타일의 차원은 fragment의 역할에 따라 다르다. `matrix_a`의 경우, 타일의 차원은 `m x k`, `matrix_b`의 경우에는 `k x n`이며, `accumulator` 타일은 `m x n`이다.

Multiplicands의 경우에는 데이터 타입 `T`는 `double`, `float`, `__half`, `__nv_bfloat16`, `char`, `unsigned char`이 될 수 있다. Accumulators의 경우에는 `double`, `float`, `int`, `__half`만 허용된다. [Element Types and Matrix Sizes](#element-types-and-matrix-sizes)에서 지원되는 accumulator와 multiplicand 타입 조합을 확인할 수 있다.

`Layout` 파라미터는 `matrix_a`와 `matrix_b` fragments에 대해서만 지정되어야 한다. `row_major` 또는 `col_major`를 지정할 수 있는데, 이름에서 알 수 있듯이 행렬 내 요소들이 메모리에서 행 또는 열 방향으로 연속적이라는 것을 가리킨다. `accumulator` matrix인 경우, `Layout` 파라미터는 `void`(default value) 이어야 하며, row or column layout은 오직 accumulator가 로드되거나 저장될 때만 지정된다 (바로 아래 함수 참조).

## `load_matrix_sync`

```c++
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
```

이 함수가 호출되면 모든 warp lanes이 `load_matrix_sync`에 도착할 때까지 기다리며, 메모리로부터 matrix fragment를 로드한다.

`mptr`는 반드시 256-bit aligned pointer 이어야 하며, 메모리 내에서 행렬의 첫 번째 요소를 가리킨다.

`ldm`은 연속된 행(for row major layout) 또는 연속된 열(for column major layout) 간 요소의 stride를 나타낸다. 요소의 타입이 `__half`라면 8의 배수이어야 하며, `float`인 경우에는 4의 배수이어야 한다.

만약 fragment가 `accumulator`라면 `layout` 인자는 `mem_row_major` 또는 `mem_col_major` 중 하나로 지정되어야 한다. `matrix_a`와 `matrix_b` fragments라면, layout은 fragment의 `layout` 파라미터로부터 추론된다.

`mptr`, `ldm`, `layout`과 `a`의 모든 템플릿 파라미터는 warp 내 모든 스레드에서 동일해야 하며, 이 함수는 warp 내 모든 스레드로부터 호출되어야 한다. 그렇지 않은 경우에는 undefined result가 발생한다.

## `store_matrix_sync`

```c++
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
```

`load_matrix_sync`와 마찬가지로, 스레드 내에서 이 함수가 호출되면 모든 warp lanes이 이 지점에 도달할 때까지 대기하며 matrix fragment를 메모리로 저장한다.

`mptr`과 `ldm`의 조건은 `load_matrix_sync`와 동일하다.

Output matrix의 layout은 `mem_row_major` 또는 `mem_col_major` 중 하나로 지정되어야 하며, `mptr`, `ldm`, `layout`과 `a`의 모든 템플릿 파라미터는 warp 내 모든 스레드에서 동일해야 한다.

## `fill_fragment`

```c++
void fill_fragment(fragment<...> &a, const T& v);
```

Matrix fragment를 상수값 `v`로 채운다. 행렬 요소로부터 각 fragment의 매핑이 지정되지 않으므로 이 함수는 일반적으로 warp 내 모든 스레드들에서 공통의 `v`값으로 호출된다.

## `mma_sync`

```c++
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

스레드 내에서 `mma_sync`를 호출하면, 모든 warp lanes이 이 지점에 도달할 때까지 대기하고, warp-synchronous matrix multiply-accumulate 연산(`D=A*B+C`)을 수행한다. `C=A*B+C` 형태의 in-place 연산도 지원된다.

`satf`와 각 matrix fragment의 템플릿 파라미터는 warp 내 모든 스레드 간에 동일해야 한다. 또한, 템플릿 파라미터 `m`, `n`, `k`는 fragments `A`, `B`, `C`, `D` 간에 일치해야 한다.

이 함수는 반드시 warp 내 모든 스레드에서 호출되어야 하며, 그렇지 않으면 undefined result를 발생시킨다.

`satf`(saturate to finite value) 모드가 `true`라면, destination accumulator에 대해 다음의 추가적인 numerical properties가 적용된다.

- If an element result is `+Infinity`, the corresponding accumulator will contain `+MAX_NORM`
- If an element result is `-Infinity`, the corresponding accumulator will contain `-MAX_NORM`
- If an element result is `NaN`, the corresponding accumulator will contain `+0`


행렬 요소로부터 각 스레드의 `fragment`로의 매핑이 지정되지 않기 때문에 각 행렬 요소는 `store_matrix_sync`를 호출한 후, 메모리(shared or global)로부터 액세스할 수 있다. Warp 내에서 모든 스레드가 모든 fragment 요소에 element-wise operation을 균일하게 적용하는 경우, 다음과 같이 `fragment`의 클래스 멤버를 통해 다음과 같이 direct element access를 구현할 수 있다.
```c++
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
/* ... */
for (int t = 0; t < frag.num_elements; t++)
    frag.x[t] *= alpha;
```

# Alternate Floating Point

Compute capability 8.0 이상의 장치에서는 아래의 타입들을 지원한다.

- `__nv_bfloat16`
- `tf32` - 이 타입의 연산에서 fragment의 요소는 `float` 타입으로 나타낸다.

# Double Precision

Compute capability 8.0 이상의 장치에서 double-precision floating point 연산을 지원한다. 이 기능을 사용하려면 `double` 타입으로 `fragment`를 사용하면 된다. `mmc_sy;nc` 연산은 .rn(rounds to nearest even) rounding modifier로 수행된다.

# Sub-byte Operations

Sub-byte WMMA 연산은 텐서 코어의 low-precision 기능에 액세스하는 방법을 제공한다. 이 기능은 preview feature이며, 제공되는 데이터 구조체 또는 API는 추후 릴리즈에서 변경될 수도 있다. 이들은 모두 `nvcuda::wmma::experimental` 네임스페이스에 정의되어 있다.

```c++
namespace experimental {
    namespace precision {
        struct u4; // 4-bit unsigned
        struct s4; // 4-bit signed
        struct b1; // 1-bit
    }
    enum bmmaBitOp {
        bmmaBitOpXOR = 1, // compute_75 minimum
        bmmaBitOpAND = 2  // compute_80 minimum
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
}
```

4-bit precision에서 사용 가능한 API는 동일하며, fragment의 데이터 타입에 `experimental::precision::u4` 또는 `experimental::precision::s4`만 지정해주면 된다. Fragment의 요소들은 함께 패킹되기 때문에 `num_storage_elements`는 `num_elements`보다 작다. 따라서, sub-byte fragment의 `num_elements` 변수는 sub-byte 타입 `element_type<T>`의 요소의 갯수를 리턴한다. 1-bit precision 또한 동일하며, `element_type<T>` to `storage_element_type<T>` 매핑은 각각 다음과 같다.
```c++
experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
experimental::precision::s4 -> int (8 elements in 1 storage element)
experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
T -> T // all other types
```

Sub-type fragment에 대해서 허용되는 layout은 `matrix_a`는 항상 `row_major`, `matrix_b`는 항상 `col_major`이다.

`load_matrix_sync`의 `ldm`의 값은 `experimental::precision::u4`와 `experimental::precision::s4`라면 32의 배수, `experimental::precision::b1`이라면 128의 배수이면 좋다.

> - `experiment::precision::u4`
> - `experiment::precision::s4`
> - `experiment::precision::b1`
>
> 위의 MMA instruction variants는 sm_90에서 deprecated 및 remove 예정

## `bmma_sync`

`mma_sync`와 역할은 동일하며, `D = (A op B) + C` 연산을 수행한다.

# Restrictions

# Element Types and Matrix Sizes

텐서 코어에서 지원하는 element types와 matrix sizes 조합은 다음과 같다.

|Matrix A|Matrix B|Accumulator|Matrix Size (m-n-k)|
|--------|--------|-----------|-------------------|
|`__half`|`__half`|`float`|16 x 16 x 16|
|`__half`|`__half`|`float`|32 x 8 x 16|
|`__half`|`__half`|`float`|8 x 32 x 16|
|`__half`|`__half`|`__half__`|16 x 16 x 16|
|`__half`|`__half`|`__half__`|32 x 8 x 16|
|`__half`|`__half`|`__half__`|8 x 32 x 16|
|`unsigned char`|`unsigned char`|`int`|16 x 16 x 16|
|`unsigned char`|`unsigned char`|`int`|32 x 8 x 16|
|`unsigned char`|`unsigned char`|`int`|8 x 32 x 16|
|`signed char`|`signed char`|`int`|16 x 16 x 16|
|`signed char`|`signed char`|`int`|32 x 8 x 16|
|`signed char`|`signed char`|`int`|8 x 32 x 16|


- **Alternate Floating Point Support:**

|Matrix A|Matrix B|Accumulator|Matrix Size (m-n-k)|
|--------|--------|-----------|-------------------|
|`__nv_bfloat16`|`__nv_bfloat16`|`float`|16 x 16 x 16|
|`__nv_bfloat16`|`__nv_bfloat16`|`float`|32 x 8 x 16|
|`__nv_bfloat16`|`__nv_bfloat16`|`float`|8 x 32 x 16|
|`precision::tf32`|`precision::tf32`|`float`|16 x 16 x 8|


- **Double Precision Support:**

|Matrix A|Matrix B|Accumulator|Matrix Size (m-n-k)|
|--------|--------|-----------|-------------------|
|`double`|`double`|`double`|8 x 8 x 4|

- **Experimental Support for Sub-byte Operations:**

|Matrix A|Matrix B|Accumulator|Matrix Size (m-n-k)|
|--------|--------|-----------|-------------------|
|`precision::u4`|`precision::u4`|`int`|8 x 8 x 32|
|`precision::s4`|`precision::s4`|`int`|8 x 8 x 32|
|`precision::b1`|`precision::b1`|`int`|8 x 8 x 128|

# Example

```c++
#include <mma.h>
using namespace nvcuda;

__global__
void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

위 커널 구현은 tensor core를 위한 wmma API를 사용하는 방법을 간단히 보여준다. 텐서 코어 구현에 대한 조금 더 자세한 내용은 [Overview of Tensor Cores](/cuda/study/21_overview_of_tensor_cores.md)에서 다루고 있다.

# References

- [NVIDIA CUDA Documentations: Warp Matrix Functions](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#warp-matrix-functions)