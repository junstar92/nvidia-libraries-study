# Table of Contents

- [Table of Contents](#table-of-contents)
- [Maximize Instruction Throughput](#maximize-instruction-throughput)
  - [Arithmetic Instructions](#arithmetic-instructions)
  - [Control Flow Instructions](#control-flow-instructions)
  - [Synchronization Instructions](#synchronization-instructions)
- [References](#references)

<br>

# Maximize Instruction Throughput

어플리케이션에서 instruction throughput을 극대화하려면

- low throughput의 arithmetic instructions 사용을 최소화해야 한다. 여기에는 최종 결과에 영향을 미치지 않도록 하는 속도와 정밀도 간의 trade-off를 포함한다. 예를 들어, regular function 대신 intrinsic function을 사용하거나, double-precision이 아닌 single-precision을 사용하거나 denormalized number를 0으로 flushing하는 것이 있다.
- control flow instructions에 의한 divergent warps를 최소화해야 한다.
- instruction의 수를 줄어야 한다. 예를 들어, [Synchronization Instructions](#synchronization-instructions)에서 언급하는 동기화 지점을 최적화하는 것이나 `__restrict__`와 같이 restricted pointers를 사용하는 것이 있다.

Instruction throughput은 멀티프로세서에서 clock cycle당 operation의 수로 정의된다. Warp의 크기가 32인 경우, 하나의 instruction은 32개의 operations에 해당한다. 따라서, clock cycle당 operation의 수가 `N`이라면, instruction throughput은 clock cycle당 `N/32` instructions가 된다.

모든 throughput은 하나의 멀티프로세서에 대한 것이므로, 전체 GPU device에 대한 처리량을 계산하려면 device의 멀티프로세서 갯수를 곱해야 한다.

## Arithmetic Instructions

> [Table 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)에서는 다양한 compute capabilities의 하드웨어에서 지원하는 native arithmetic instruction의 처리량을 나열한다.

이외의 instruction이나 함수들은 [Table 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)에 나열된 native instruction으로 구현된다. Compute capabilities마다 구현은 다를 수 있으며, 컴파일러 버전에 따라서 컴파일 이후의 native instructions의 수가 다를 수도 있다. `cuobjdump`를 통해 `cubin` object에서의 특정 구현을 확인해볼 수 있다.

일반적으로 `-ftz=true` 옵션으로 컴파일된 코드는 `-ftz=false`로 컴파일된 코드보다 성능이 더 좋은 경향이 있다 (denormalized numbers are flushed to zero). 마찬가지로, `-prec-div=false`(less precise division)로 컴파일된 코드는 `-prec-div=true`로 컴파일된 코드보다 성능이 더 좋은 경향이 있으며, `-prec-sqrt=false`(less precise square root)로 컴파일된 코드는 `-prec-sqrt=true`로 컴파일된 코드보다 성능이 더 좋은 경향이 있다.

아래의 arithmetic instructions에 대해서 설명하고 있으므로, 필요하다면 [공식 문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)를 참조 바람.

- Single-Precision Floating-Point Division
- Single-Precision Floating-Point Reciprocal Square Root
- Single-Precision Floating-Point Square Root
- Sine and Cosine
- Integer Arithmetic
- Half Precision Arithmetic
- Type Conversion

## Control Flow Instructions

모든 control flow instruction(`if`, `switch`, `do`, `for`, `while`)은 동일한 warp 내 스레드들이 divergent될 수 있기 때문에 instruction throughput에 상당한 영향을 미칠 수 있다 (warp divergence를 발생시킴). 만약 warp divergence가 발생한다면, 서로 다른 execution path들은 순차적으로 수행되어 해당 warp 내 실행되는 instruction의 수를 증가시킨다.

Control flow가 thread ID에 의존적인 경우, 최상의 성능을 얻으려면 **divergent warps**의 수를 최소화하도록 조건을 작성해야 한다. 이는 블록 내 warp의 분포가 결정적(diterministic)이기 때문에 가능하다. 만약 제어문이 (`threadIdx / warpSize`)에만 의존적이라면, 한 warp 내에서의 조건문의 결과는 항상 같으므로 어떠한 warp divergence도 발생하지 않는다.

> Warp divergence에 대한 내용은 아래 포스트에서 조금 더 자세히 다루고 있다.
> - [Understanding Warp Execution](/cuda/study/06_understanding_warp_execution.md)
> - [Avoiding Branch Divergence](/cuda/study/07_avoiding_branch_divergence.md)

때때로, 컴파일러는 루프를 풀거나(unroll loops), branch prediction을 사용하여 짧은 `if`나 `switch` 블록을 최적화한다. 이 경우에는 어떠한 warp도 divergent하지 않는다.

> 프로그래머가 `#pragma unroll` 지시문을 사용하여 loop unrolling을 직접 컨트롤할 수 있다.

Branch prediction을 사용할 때, 제어 조건에 따라 달라지는 instruction은 스킵되지 않는다. 대신, 스레드마다 해당 조건문이 true 또는 false인지 예측하며 각 instruction이 실행을 위해 스케쥴링되지만 true인 경우에만 실제로 실행된다. False인 instruction은 결과를 실제로 write하지 않으며, 주소를 평가하거나 피연산자를 읽지 않는다.

## Synchronization Instructions

`__syncthreads()`의 처리량은 compute capability에 따라서 다르며, 각 compute capability에서의 처리량은 다음과 같다.

- compute capability 6.0 : 32 operations per clock cycle
- compute capability 7.x, 8.x : 16 operations per clock cycle
- compute capability 5.x, 6.1, 6.2 : 64 operations per clock cycle

> `__syncthreads()`는 멀티프로세서를 idle 상태로 만들 수 있기 때문에 성능에 상당한 영향을 미칠 수 있다.

<br>

# References

- [NVIDIA CUDA Documentations: Maximize Instruction Throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-instruction-throughput)