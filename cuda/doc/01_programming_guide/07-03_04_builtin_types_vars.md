# Table of Contents

- [Table of Contents](#table-of-contents)
- [Built-in Vector Types](#built-in-vector-types)
  - [char, short, int, long, long long, float, double](#char-short-int-long-long-long-float-double)
  - [dim3](#dim3)
- [Built-in Variables](#built-in-variables)
  - [gridDim](#griddim)
  - [blockIdx](#blockidx)
  - [blockDim](#blockdim)
  - [threadIdx](#threadidx)
  - [warpSize](#warpsize)
- [References](#references)

<br>

# Built-in Vector Types

## char, short, int, long, long long, float, double

기본 integer와 floating-point 타입으로부터 파생되는 벡터 타입이 있다. 이러한 벡터 타입은 구조체(structure)이며, 각 컴포넌트들을 `x`,`y`,`z`,`w` 필드로 액세스할 수 있다. 또한, `make_<type name>` 형태의 생성자 함수로부터 생성할 수 있다.
```c++
int2 make_int2(int x, int y);
```
예를 들면, 위와 같은 생성자 함수가 있으며, `int2` 타입의 벡터는 `(x, y)` 값을 갖는다.

> 각 벡터 타입에 대한 alignment requirements는 [Table 5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types-alignment-requirements-in-device-code)를 참조

## dim3

`dim3`는 `uint3`을 베이스 클래스로 하는 정수 벡터 타입이며, 차원을 지정하는데 주로 사용된다. `dim3` 타입 변수를 정의할 때, 값이 지정되지 않으면 1로 초기화된다.

<br>

# Built-in Variables

내장 변수는 grid와 block 차원과 block과 thread의 인덱스를 나타내는데 사용된다. 이들은 오직 device에서 수행되는 함수 내에서만 유효하다.

## gridDim

`dim3` 타입의 변수이며 grid의 차원을 나타낸다.

## blockIdx

`uint3` 타입의 변수이며 grid 내 block index를 나타낸다.

## blockDim

`dim3` 타입의 변수이며 block의 차원 크기를 나타낸다.

## threadIdx

`uint3` 타입의 변수이며 block 내 thread index를 나타낸다.

## warpSize

`int` 타입의 변수이며 스레드 내에서 warp size를 나타낸다.

<br>

# References

- [NVIDIA CUDA Documentations: Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types)
- [NVIDIA CUDA Documentations: Built-in Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables)