# Table of Contents

- [Table of Contents](#table-of-contents)
- [Instruction Optimization](#instruction-optimization)
- [Arithmetic Instructions](#arithmetic-instructions)
  - [Division Modulo Operations](#division-modulo-operations)
  - [Loop Counters Signed vs. Unsigned](#loop-counters-signed-vs-unsigned)
  - [Reciprocal Square Root](#reciprocal-square-root)
  - [Other Arithmetic Instructions](#other-arithmetic-instructions)
  - [Exponentiation With Small Fractional Arguments](#exponentiation-with-small-fractional-arguments)
  - [Math Libraries](#math-libraries)
  - [Precision-related Compiler Flags](#precision-related-compiler-flags)
- [Memory Instructions](#memory-instructions)
- [References](#references)

<br>

# Instruction Optimization

Instruction이 실행되는 방법을 잘 알고 있으면, 소위 프로그램 내에서 hot spot이라고 불리는 빈번하게 실행되는 코드에 대해 low-level optimizations을 적용할 수 있다. 일반적인 best practices는 모든 high-level 최적화를 완료한 이후에 low-level 최적화 적용을 추천한다.

<br>

# Arithmetic Instructions

단정밀도(single-precision) 부동소수점에서 최고의 성능을 제공하며, 이를 사용하는 것이 좋다. 각 산술 연산의 처리량은 CUDA C++ Programming Guide의 [Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)에서 확인할 수 있다.

## Division Modulo Operations

> 비용이 큰 division과 modulo 계산 대신 shift 연산을 사용하는 것이 좋다

Integer에 대한 division과 modulo 연산은 비용이 특히 크며, 가능한 bitwise 연산으로 대체하는 것이 좋다.

만약 `n`이 2의 배수라면, `(i/n)`은 `(i>>log2(n))`과 동일하고 `(i%n)`은 `(i & (n-1))`과 동일하다. 컴파일러는 `n`이 **literal**이라면 이 변환을 알아서 수행한다.

## Loop Counters Signed vs. Unsigned

> Loop counter로 unsigned integer보다 signed integer를 사용하는 것이 좋다

C 언어 표준에서 unsigned integer overflow semantics는 잘 정의되어 있지만, signed integer overflow는 undefined results를 발생시킨다. 따라서, 컴파일러는 부호가 있는 산술에 대해 더 공격적으로 최적화할 수 있다. 루프의 카운터에서는 더욱 중요하다. 카운터는 항상 양수인 것이 일반적이므로 카운터를 부호가 없는 타입으로 선언하고 싶을 수 있지만, 약간 더 좋은 성능을 위해서는 부호가 있는 타입으로 선언해야 한다.

예를 들어, 아래의 코드를 고려해보자.
```c++
for (i = 0; i < n; i++) {
    out[i] = in[offset + stride * i];
}
```
위 코드에서 `stride * i`가 32비트 정수에 대해 오버플로우가 발생할 수 있으므로, `i`가 unsigned로 선언된다면 overflow semantics는 컴파일러가 strength reduction과 같은 일부 최적화를 적용하지 못하도록 한다. `i`를 signed로 선언하는 경우에는 overflow semantics가 정의되지 않고, 컴파일러가 최적화를 적용할 여유가 더 많이 생기게 된다.

## Reciprocal Square Root

Reciprocal square root (제곱근 역수)는 단정밀도의 경우에는 `rsqrtf()`, 배정밀도인 경우에는 `rsqrt()`로 항상 명시적으로 호출해야 한다. 컴파일러는 IEEE-754를 위반하지 않는 경우에만 `1.0f/sqrtf(x)`를 `rsqrtf(x)`로 최적화한다.

## Other Arithmetic Instructions

> double -> float로의 automatic conversion은 피하는 것이 좋다

컴파일러는 경우에 따라 conversion instruction을 삽입해야 하며, 이로 인해 추가적인 execution cycles이 발생하게 된다. 이에 해당하는 경우는 아래와 같다.

- `char` 또는 `short`에 대해 연산하는 함수의 피연산자가 `int`로 변환되어야 할 필요가 있는 경우 (어떤 의미인지 정확히 파악은 되지 않음..)
- 단정밀도 계산의 입력으로 사용되는 배정밀도 상수(0.123과 같이 타입 접미사없이 정의되는 경우)

단정밀도 코드에서는 `float` 타입과 single-precision math functions를 사용하는 것을 매우 권장한다.

CUDA math 라이브러리의 complementary error function인 `erfcf()`는 full single-precision 정확도이면서 매우 빠르다.

## Exponentiation With Small Fractional Arguments

Fractional exponents(분수 지수)를 사용하는 경우, 제곱근, `pow()`를 사용하는 것과 비교했을 때 세제곱근 및 이들의 역수를 사용하여 거듭제곱을 크게 가속할 수 있다. 이를 사용하면, 지수가 `1/3`과 같이 부동소수점 수로 정확히 표현할 수 없는 경우에는 `pow()`를 사용하면 초기 representational error를 크게 만드므로, 훨씬 더 정확한 결과를 제공할 수도 있다.

[Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#id8)는 분수로 표현되는 몇 가지 지수들에 대한 공식을 나타내며, 표에 나타난 공식들은 `x >= 0, x != -0`일 때, 즉, `signbit(x) == 0`일 때에만 유효하다.

## Math Libraries

> 정밀도보다 속도가 중요한 경우, fast math library를 사용하면 좋다.

두 가지 타입의 런타임 math operations를 제공하는데, 이들은 이름으로 구분할 수 있다. 수학 연산들 중에는 이름 처음에 두 개의 밑줄이 붙어 있는 것이 있고 아닌 것이 있다 (e.g., `__functionName()`과 `functionName()`). `__functionName()`과 같이 함수 이름이 두 개의 밑줄로 시작하는 것들은 하드웨어 레벨에 직접 매핑된다. 속도는 더 빠르지만 다소 낮은 정확도를 제공한다 (e.g., `__sinf(x)` 또는 `__expf(x)`). 반면 밑줄이 없는 `functionName()`과 같은 이름의 함수들은 더 느리지만 더 높은 정확도를 제공한다 (e.g., `sinf(x)` 또는 `expf(x)`). `__sinf(x)`, `__cosf(x)`, `__expf(x)` 함수의 처리량은 `sinf(x)`, `cosf(x)`, `expf(x)`의 처리량보다 훨씬 높다. 후자의 함수들의 경우, 인자 `x`의 크기를 감소시켜야 하는 경우에는 비용이 더욱 크다 (1챠적으로 느리게 동작한다). 또한, 이러한 경우에는 인자 크기를 감소시키는 코드에서 local memory를 사용하므로 local memory의 높은 latency로 인해 성능이 더욱 저하될 수 있다.

또한 같은 인수를 사용하는 사인과 코사인을 계산할 때마다 성능을 최적화하려면 `sincos` instruction 계열을 사용해야 한다.

- `__sincosf()` for single-precision fast math
- `sincosf()` for regular single-precision
- `sincos()` for double-precision

`nvcc`의 컴파일 옵션인 `-use_fast_math`를 사용하는 것은 모든 `functionName()`의 호출을 동일한 `__functionName()`으로 호출로 강제한다. 또한 single-precision denormal 지원을 비활성화하고 일반적인 single-precision의 정밀도를 낮춘다. 이는 숫자의 정밀도를 낮추고 특수한 경우를 변경할 수 있는 공격적인 최적화이다. 따라서, 성능(속도) 향상과 동작의 변경이 허용되는 경우에만 내장 함수 호출을 선택적으로 도입하는 것이 좋다. `-use_fast_math`는 single-precision 부동소수점에 대해서만 유효하다.

작읍 거듭제곱(x2 또는 x3)의 경우에는 그냥 곱셈이 `pow()`와 같은 일반 지수 루틴보다 거의 대부분 더 빠르다. 컴파일러 최적화 개선을 통해 이 부분의 차이를 줄이려고 지속적으로 노력하지만, 명시적인 곱셈(또는 전용 인라인 함수와 매크로)이 상당한 이점을 가질 수 있다. 이는 동일한 밑(base)에 대한 거듭제곱이 필요한 경우에 common sub-expression elimination(CES) 최적화에서 컴파일러를 도와주므로 상당한 이점을 가질 수 있다.

밑이 2나 10인 거듭제곱인 경우에는 `pow()` 또는 `powf()` 함수 대신 `exp2()` 또는 `expf2()`, `exp10()` 또는 `expf10()`을 사용하는 것이 좋다. `pow()`와 `powf()` 함수는 다양한 경우의 수를 고려하므로 register pressure와 instruction count 측면에서 매우 무거운 함수이며 밑과 지수의 전체 범위에서 좋은 정확도를 달성할 수 어렵다. 반면, `exp2()`, `exp2f()`, `exp10()`, `expf10()`은 성능면에서 `exp()` 및 `expf()`와 유사하면서도 `pow()`/`powf()` 보다보다 10배 더 빠를 수 있다.

지수가 `1/3`인 거듭제곱인 경우에는 일반 지수함수인 `pow()` 또는 `powf()` 대신 `cbrt()` 또는 `cbrtf()` 함수를 사용하는 것이 훨씬 더 빠르다. 마찬가지로 지수가 `-1/3`인 경우에도 `rcbrt()` 또는 `rcbrtf()`를 사용하는 것이 좋다.

`sin(π*<expr>)` 대신 `sinpi(<expr>)`을, `cos(π*<expr>)` 대신 `cospi(<expr>)`을, 그리고 `sincos(π*<expr>)` 대신 `sincospi(<expr>)`로 바꾸는 것이 좋다. 이는 정확도와 성능면에서 모두 유리하다. 예를 들어, 라디안이 아닌 도 단위로 사인 함수를 평가하려면, `sinpi(x/180.0)`를 사용하는 것이 좋다. 비슷하게, 단정밀도 함수인 `sinf()`, `cosf()`, `sincosf()`는 `sinpif()`, `cospif()`, `sincospif()`로 대체하는 것이 좋다. `sinpi()`가 `sin()`보다 성능상 더 좋은 이유는 간서화된 argument reduction 때문이다. 정확도 측면어세도 `sinpi()`가 $\pi$를 명시적으로 곱하지 않고, 실제대로 무한 정밀도(infinitely precise)인 π를 사용하기 때문이다.

## Precision-related Compiler Flags

기본적으로 `nvcc`는 IEEE-compliant code를 생성하지만, 다소 부정확하지만 속도가 더 빠른 코드를 생성할 수 있는 컴파일 옵션을 제공한다.

- `-ftz=true` (denormalized numbers are flushed to zero)
- `-prec-div=false` (less precise division)
- `-prec-sqrt=false` (less precise square root)

조금 더 공격적인 최적화로는 `-use_fase_math` 옵션을 사용하는 것이 있다. 이를 사용하면, 위에서 언급했듯이 `functionName()` 호출을 동일한 `__functionName()` 호출로 강제한다. 이렇게 하면 정밀도와 정확도가 떨어질 수 있지만 더 빠르게 코드가 실행된다.

<br>

# Memory Instructions

> Global memory의 사용을 최소화해야 하며, 가능한 곳마다 shared memory를 사용하는 것이 좋다.

Memory instruction에는 shared, local, global memory에 write/read 하는 모든 instrunction이 포함된다. 캐시되지 않은 local 또는 global memory에 액세스할 때는 수백 사이클의 memory latency가 발생한다.

예를 들어, 아래 샘플 코드의 할당 연산자는 처리량은 높지만, global memory로부터 데이터를 읽는데 수백 사이클의 latency가 발생하게 된다.
```c++
__shared__ float shared[32];
__device__ float device[32];
shared[threadIdx.x] = device[threadIdx.x];
```

Global memory latency는 독립적인 산술 instruction이 있는 경우에 스레드 스케줄러에 의해서 hiding 될 수 있다. 그러나 가능하면 global memory에 액세스하지 않는 것이 가장 좋다.

<br>

# References

- [NVIDIA CUDA Documentation: Instruction Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#instruction-optimization)