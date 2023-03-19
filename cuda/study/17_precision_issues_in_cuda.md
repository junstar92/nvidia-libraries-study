# Table of Contents

- [Table of Contents](#table-of-contents)
- [Precision Issues in Matrix Multiplication Example](#precision-issues-in-matrix-multiplication-example)
- [Rounding](#rounding)
- [Fused Multiply-Add Operation](#fused-multiply-add-operation)
- [References](#references)

<br>

# Precision Issues in Matrix Multiplication Example

행렬 곱셈을 구현한 [matmul.cu](/cuda/code/matmul/matmul.cu)의 코드를 살펴보면, GPU에서 연산한 결과를 CPU에서 연산한 결과를 비교할 때, 각 요소의 차이가 1.e-5f보다 크면 GPU 연산 결과가 잘못되었다고 판단하고 있다.
```c++
void checkResult(float const* host_ref, float const* gpu_ref, int const m, int const n)
{
    float eps = 1.e-5f;
    unsigned int idx;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            idx = n * r + c;
            if (fabsf(host_ref[idx] - gpu_ref[idx]) > eps) {
                std::cout << std::fixed << std::setprecision(6)
                    << "Result verification failed at (" << r << "," << c << ") host: " << host_ref[idx] << " gpu: " << gpu_ref[idx] << "\n";
                return;
            }
        }
    }
}
```

CUDA Sample Code([matrixMul](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul))에서는 위와 같이 두 연산 결과값의 차이가 아닌 실제 결과값 대비 에러율을 계산하여 이 값이 1.e-6보다 큰지 체크하여 GPU에서의 결과를 검증하고 있다.

방법은 조금 다르지만, 결과적으로 GPU에서의 행렬 곱셈 연산 결과가 CPU에서의 연산 결과와 정확히 같지는 않다는 것을 알 수 있다. 실제로 [matmul.cu](/cuda/code/matmul/matmul.cu)의 `checkResult` 함수에서 `eps`의 값을 `1.e-6f`로 바꿔서 실행시켜보면 아래와 같이 GPU와 CPU의 연산 결과가 다르다고 출력하는 것을 볼 수 있다.
```
$ ./matmul
> Starting matrix multiplication at device 0: NVIDIA GeForce RTX 3080
> Matrix A  : (1024 x 1024)
> Matrix B  : (1024 x 1024)
> BLOCK_SIZE: 32
matmulHost      : 3334.39 ms
matmulNaive     : 1.35434 ms
Result verification failed at (0,11) host: 16.960281 gpu: 16.960283
matmulSmem      : 0.990208 ms
Result verification failed at (0,11) host: 16.960281 gpu: 16.960283
```

위와 같은 문제는 부동소수점의 가수부(fraction)가 표현할 수 있는 수가 제한적이기 때문이다. 따라서, 특정 소수점 아래까지 내려가게 되면 표현하고자 하는 수를 정확히 표현하지 못할 수 있다. 지수부(exponent)에 의해서도 동일한 문제가 발생할 수 있다.

즉, floating-point의 정밀도에는 한계가 있으며, 이는 **limited significant figures** (유효숫자) 때문이다.

아래 코드는 이러한 전형적인 정밀도 문제를 잘 보여주는 예제 코드이다. 여기서 각 배열 요소에는 `2^32=8388608`에 소수점 첫재 자리의 수만 다른 값이 입력된다.
```c++
#include <stdio.h>

int main(int argc, char** argv)
{
    float f[10];
    // 2^32 = 8,388,608
    f[0] = 8388608.0;
    f[1] = 8388608.1;
    f[2] = 8388608.2;
    f[3] = 8388608.3;
    f[4] = 8388608.4;
    f[5] = 8388608.5;
    f[6] = 8388608.6;
    f[7] = 8388608.7;
    f[8] = 8388608.8;
    f[9] = 8388608.9;

    for (int i = 0; i < 10; i++)
        printf("f[%d]: %f\n", i, f[i]);
}
```

위 코드를 실행한 결과는 다음과 같다.
```
f[0]: 8388608.000000
f[1]: 8388608.000000
f[2]: 8388608.000000
f[3]: 8388608.000000
f[4]: 8388608.000000
f[5]: 8388608.000000
f[6]: 8388609.000000
f[7]: 8388609.000000
f[8]: 8388609.000000
f[9]: 8388609.000000
```
분명 소수점 첫째 자리에 0이 아닌 값을 지정했으나, 소수 부분이 모두 0으로 출력된다. 그리고 여섯 번째 수까지는 `8388608.000000`으로 출력되고, 그 이후는 `8388608.000000`으로 출력되는 것을 볼 수 있다. 이를 통해 유효 숫자에 의해서 부동소수점이 표현할 수 있는 수에는 한계가 있다는 것을 알 수 있다.

<br>

# Rounding

이렇게 유효 숫자에 의해서 표현할 수 없는 숫자들은 `rounding`(반올림)을 통해 표현할 수 있는 숫자로 바뀐다. 단, 이러한 rounding의 방법(올림, 내림, 또는 반올림 등)은 하드웨어의 설계에 따라 다르다. 행렬 곱셈 예제 코드에서 CPU와 GPU의 결과가 다른 것도 rounding 전략이 하드웨어에 따라 다르기 때문이다.

기본적인 rounding 방법에는 아래와 같은 방법들이 있다.

- `RN` : round to the nearest floating-point number (ties to even)
- `RD` : round towards $-\infty$
- `RU` : round towards $+\infty$
- `RZ` : round towards zero

> `IEEE 754`에는 `round to the nearest`, `ties away from zero` 방법도 정의하고 있음

이러한 rounding으로부터 발생하는 error를 해결하는 방법 중, 가장 흔히 사용되는 방법은 error bound를 설정하는 것이다. 즉, 일정 소수점 자리 이하는 모두 0이라고 생각하고 error가 발생하더라도 무시한다. 일반적으로 single floating-point에서는 대략 소수 6번째 자리까지의 정밀도를 가지고 있기 때문에 error bound를 `1.e-5f`로 설정하면 된다.

또 다른 방법으로는 연산할 때 round 방법을 직접 설정하는 것이 있다. CUDA 문서([Table 10](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library))를 살펴보면, 각 연산들에 대해 round 방법을 지정할 수 있는 내장 함수들을 제공한다. 예를 들어, `__fadd_rn(x, y)`와 같이 사용하여 덧셈 연산에 대해 `RN` 방법을 지정할 수 있다.

<br>

# Fused Multiply-Add Operation

일반적으로 부동소수점 연산은 느리다. 따라서, 각 가속기에서는 이러한 부동소수점 연산 속도를 높이기 위해 여러 가지 방법들을 제공한다. NVIDIA GPU의 경우에는 **FMA(Fused Multiply-Add) Operationt** 을 제공한다.

> 모든 CUDA compute 장치들은 binary floating-point 표현에 대해 `IEEE 754` 표준을 준수하는데, 몇 가지 예외는 있다. 이러한 예외 때문에 CPU 연산 결과와 차이가 발생하게 되는 것이다. 주요한 차이점 중 하나가 바로 **FMA instruction** 이다. FMA instruction은 multiply-add operation을 결합하여 하나의 instruction으로 수행한다.

일반적으로 multiply and add 연산(`X * Y + Z`)은 아래의 과정을 통해 계산된다.
```
RN(RN(X * Y) + Z)
```
이 연산에는 두 번의 `RN` round가 포함되는데, round 연산은 cost가 크기 때문에 이렇게 두 번 적용하는 것은 비효율적이다.

반면, `FMA` 연산은 multiply and add 연산을 한 번에 수행한다. 따라서, 곱셈 및 덧셈 연산을 한 번에 수행한 뒤, 한 번의 round를 적용한다.
```
RN(X * Y + Z)
```

FMA 연산은 컴파일러에 따라서 다르게 적용될 수 있으며, 릴리즈/디버그 모드에 따라서도 다를 수 있다. 또한, `fast math`와 관련된 컴파일 옵션에 따라서도 다를 수 있다.

`nvcc` 컴파일러에서는 `-fmad=false` 옵션을 지정하면 FMA 연산을 사용하지 않도록 할 수 있으며, 기본값은 `true`이다. `nvcc`에서 `fast math`와 관련된 컴파일 옵션은 `--use_fast_math`이며, 이 옵션을 지정하는 것을 아래의 옵션을 지정하는 것과 동일하다.
```
-ftz-true --prec-div=false --prec-sqrt=false --fmad=true
```

행렬 곱셈 예제 코드를 컴파일할 때, `--fmad=false` 컴파일 옵션을 지정하여 컴파일하면 CPU와 GPU의 결과가 완전히 동일하게 나오는 것을 확인할 수 있다.

> CUDA에서 numerical accuracy와 precision에 관련한 내용은 [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyIsIm5jaWQiOiJyZWYtZGV2LTUwMDQwOSJ9)에서 자세히 다루고 있다.

<br>

# References

- [NVIDIA CUDA Documentation: Floating-Point Standard](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#floating-point-standard)
- [NVIDIA CUDA Documentation: Numerical Accuracy and Precision](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numerical-accuracy-and-precision)
- [NVIDIA CUDA Documentation: Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix)
- [Youtube: CUDA 프로그램 정밀도 문제](https://www.youtube.com/watch?v=IJ_k8SCR3Y8&ab_channel=HPCLab.KOREATECH)