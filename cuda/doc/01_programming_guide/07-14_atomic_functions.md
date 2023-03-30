# Table of Contents

- [Table of Contents](#table-of-contents)
- [Atomic Functions](#atomic-functions)
  - [Arithmetic Functions](#arithmetic-functions)
    - [atomicAdd()](#atomicadd)
    - [atomicSub()](#atomicsub)
    - [atomicExch()](#atomicexch)
    - [atomicMin()](#atomicmin)
    - [atomicMax()](#atomicmax)
    - [atomicInc()](#atomicinc)
    - [atomicDec()](#atomicdec)
    - [atomicCAS()](#atomiccas)
  - [Bitwise Functions](#bitwise-functions)
    - [atomicAnd()](#atomicand)
    - [atomicOr()](#atomicor)
    - [atomicXor()](#atomicxor)
- [References](#references)

<br>

# Atomic Functions

Atomic function은 global memory 또는 shared memory에서 32-bit or 64-bit word에 대해 **read-modify-write** 연산을 수행한다. `float2`나 `float4` 벡터 타입의 경우에는 각 요소마다 read-modify-write opration이 수행된다.

예를 들어, `atomicAdd()`는 global 또는 shared memory의 어느 주소에서 word 하나를 읽고, 그 값에 더한 뒤, 다시 동일한 위치의 주소에 결과를 저장한다.

> Atomic function은 device function에서만 사용할 수 있다.

아래에서 설명하는 atomic function은 `cuda::memory_order_relaxed` 순서를 가지며, 특정 scope에서만 atomic이다.

- `_system`라는 접미사가 붙은 atomic APIs(ex, `__atomicAdd_system`)의 scope는 `cuda::thread_scope_system`이다.
- 어떠한 접미사도 붙지 않은 atomic APIs(ex, `__atomicAdd`)의 scope는 `cuda::thread_scope_device`이다.
- `_block` 접미사가 붙은 atomic APIs(ex, `__atomicAdd_block`)의 scope는 `cuda::thread_scope_block`이다.

아래 예제 코드는 CPU와 GPU 모두에서 `addr` 주소에 정수 값을 atomic으로 더하는 것을 보여준다.
```c++
__global__ void myKernel(int* addr) {
    atomicAdd_system(addr, 10); // only avaiable on devices with compute capability 6.x
}

void foo() {
    int* addr;
    cudaMallocManaged(&addr, 4);
    *addr = 0;

    myKernel<<<...>>>(addr);
    __sync_fetch_and_add(addr, 10); // CPU atomic operation
}
```

CUDA에서 지원하는 atomic 연산 이외의 연산은 `atomicCAS()`(Compare and Swap)으로 구현할 수 있다. 예를 들어, 배정밀 부동소수점 타입에 대한 `atomicAdd()`는 compute capability 6.0보다 낮은 device에서는 지원되지 않는데, 이 경우에는 `atomicCAS()`를 통해 구현할 수 있다.
```c++
#if __CUDA_ARCH__ < 600
__device__
double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparision to avoid hang in case of NaN
        // (since Nan != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
```

Device-wide atomic APIs뿐만 아니라 system-wide, block-wide atomic APIs도 지원하며, 접미사만 다를 뿐 기본적인 동작, 사용 방법은 동일하다.

> Compute capability 6.0 미만의 device에서는 device-wide atomic operations만 지원한다.

> Compute capability 7.2 미만의 `tegra` device에서는 system-wide atomic operations를 지원하지 않는다.

## Arithmetic Functions

### atomicAdd()

```c++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
__nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
__nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val);
float2 atomicAdd(float2* address, float2 val);
float4 atomicAdd(float4* address, float4 val);
```

지원되는 `atomicAdd()`는 위와 같다. `atomicAdd()`는 global 또는 shared memory의 `address` 주소에서 16-bit, 32-bit, 또는 64-bit의 `old` 값을 읽고, `(old + val)`을 수행한 뒤, 이 값을 다시 동일한 주소의 메모리에 저장한다. 수행되는 3가지 연산이 하나의 atomic transaction으로 수행된다. 그리고 `old`값을 리턴한다.

> 32-bit floating-point version의 `atomicAdd()`는 compute capability 2.x 이상의 device에서만 지원한다.

> 64-bit floating-point version의 `atomicAdd()`는 compute capability 6.x 이상의 device에서만 지원한다.

> 32-bit `__half2` floating-point version의 `atomicAdd()`는 compute capability 6.x 이상의 device에서만 지원한다. `__half2` 또는 `__nv_bfloat162`의 add operation에 대한 atomicity는 두 개로 분리된 `__half` 또는 `__nv_bfloat16` 요소 각각에 대해 개별적으로 보장되며, 단일 32-bit 액세스로의 atomic은 보장되지 않는다. 

> `float2`와 `float4` floating-point vector version의 `atomicAdd()`는 compute capability 9.x 이상의 device에서만 지원된다. `__half2`의 경우와 동일하게 `float2`와 `float4`에서의 add operation은 각각 2개 또는 4개의 요소에 대해 개별적으로 atomicity가 보장되며, 하나의 64-bit 또는 128-bit 액세스로의 atomic은 보장되지 않는다.
>
> 또한, global memory 주소에 대해서만 지원한다.

> 16-bit `__half` floating-point version의 `atomicAdd()`는 compute capability 7.x 이상의 device에서만 지원한다.

> 16-bit `__nv_bfloat16` floating-point version의 `atomicAdd()`는 ompute capability 8.x 이상의 device에서만 지원한다.

### atomicSub()

```c++
int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit word `old`값을 읽고, `(old - val)`을 수행한 뒤, 결과값을 다시 동일한 메모리 주소에 저장한다. 마찬가지로 3개 연산이 하나의 atomic transaction으로 수행되며, `old`값을 반환한다.

### atomicExch()

```c++
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `val`을 동일한 메모리 주소에 저장한다. 2개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

### atomicMin()

```c++
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `old`와 `val` 중 최솟값을 계산한 뒤, 그 값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> 64-bit version의 `atomicMin()`은 compute capability 5.0 이상의 device에서만 지원한다.

### atomicMax()

```c++
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `old`와 `val` 중 최댓값을 계산한 뒤, 그 값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> 64-bit version의 `atomicMax()`은 compute capability 5.0 이상의 device에서만 지원한다.

### atomicInc()

```c++
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit word `old`값을 읽고, `((old >= val) ? 0 : (old + 1))`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> `atomicInc()`를 사용한 예제 코드를 [reduce_fence.cu](/cuda/code/reduce_integer/reduce_fence.cu)에서 확인해볼 수 있다 (single-pass reduction).

### atomicDec()

```c++
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit word `old`값을 읽고, `(((old == 0) || (old > val)) ? val : (old - 1))`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

### atomicCAS()

Compare and Swap 연산이다.

```c++
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address,
                             unsigned short int compare,
                             unsigned short int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `((old == compare) ? val : old)`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

## Bitwise Functions

### atomicAnd()

```c++
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `(old & val)`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> 64-bit version의 `atomicAnd()`는 compute capability 5.0 이상의 device에서만 지원한다.

### atomicOr()

```c++
int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `(old | val)`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> 64-bit version의 `atomicOr()`는 compute capability 5.0 이상의 device에서만 지원한다.

### atomicXor()

```c++
int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                unsigned long long int val);
```

Global 또는 shared memory의 `address` 주소에서 32-bit 또는 64-bit word `old`값을 읽고, `(old ^ val)`을 계산한 뒤, 결과값을 동일한 메모리 주소에 저장한다. 3개 연산이 하나의 atomic transaction으로 수행되며, `old`를 반환한다.

> 64-bit version의 `atomicXor()`는 compute capability 5.0 이상의 device에서만 지원한다.

<br>

# References

- [NVIDIA CUDA Documentations: Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)