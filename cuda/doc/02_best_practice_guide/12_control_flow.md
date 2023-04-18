# Table of Contents

- [Table of Contents](#table-of-contents)
- [Branching and Divergence](#branching-and-divergence)
- [Branch Predication](#branch-predication)
- [References](#references)

<br>

# Branching and Divergence

> 최대한 동일한 warp에서 다른 execution paths를 피하는 것이 좋다.

`if`, `switch`, `do`, `for`, `while`과 같은 flow control instructions는 동일한 warp 내 스레드들을 분산시켜 instruction throughput에 상당한 영향을 미칠 수 있다. 다시 말하자면, 각 flow control에서 스레드들이 서로 다른 경로를 실행하는 경우를 의미한다. 이 경우에는 다른 경로들을 별도로 각각 실행하게 되며, 해당 warp에 대해서 실행되는 총 instruction의 수가 증가하게 된다.

Control flow가 스레드 ID에 의존하는 경우, 최상의 성능을 얻으려면 divergent warps의 수가 최소화하도록 조건을 작성해야 한다.

이것이 가능한 이유는 SIMT 아키텍처에서 블록 전체의 warp 분포가 결정론적(deterministic)이기 때문이다. 전형적인 예시로 제어 조건이 `(threadIdx / WSIZE)`에만 의존하는 경우가 이에 해당하며, 여기서 `WSIZE`는 warp의 크기이다.

적은 수의 instruction만을 포함하는 분기의 경우, warp divergence는 일반적으로 약간의 성능 손실을 초래한다. 예를 들어, 컴파일러는 predication을 사용하여 실제 발생하는 분기를 피할 수 있다. 대신 모든 instruction이 스케줄링되지만, 스레드별 condition code 또는 predicate가 instruction을 실행할 스레드를 제어한다. **False predicate**가 있는 스레드는 결과를 쓰지 않고, 주소를 평가하지 않거나 피연산자를 읽지 않는다.

Volta 아키텍처부터 Independent Thread Scheduling를 통해, warp는 데이터 종속적인 조건 블록 외부에서도 분기된 채로 있을 수 있다. 이때, 명시적인 `__syncwarp()`를 사용하여 이어지는 명령을 위해 warp가 다시 수렴되었음을 보장할 수 있다.

<br>

# Branch Predication

> 컴파일러가 루프나 제어문 

경우에 따라 컴파일러는 branch prediction을 대신 사용하여 loops를 풀거나(unroll loop), `if` 또는 `switch`문을 최적화할 수 있다. 이 경우에는 어떠한 warp도 발산할 수 없다. 프로그래머는 아래와 같이 작성하여 loop unrolling을 제어할 수도 있다.
```c++
#pragma unroll
```

Branch prediction을 사용하는 경우, 제어 조건에 따라 실행이 달라지는 instruction을 건너뛰지는 않는다. 대신, 이러한 각 instruction은 제어 조건에 따라 true 또는 false로 설정되는 스레드별 조건 코드와 연관된다. 이와 같은 instruction들은 실행하도록 스케줄링은 되어 있지만 true predicate가 있는 instruction만 실제로 실행된다. False predicate의 instruction들은 결과를 write하지 않으며 주소(addresses)를 평가하거나 피연산자를 읽지 않는다.

컴파일러는 branch condition으로 제어되는 instruction의 수가 특정 임계값보다 작거나 같은 경우에만 branch instruction을 predicated instruction으로 대체한다.

<br>

# References

- [NVIDIA CUDA Documentation: Control Flow](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#control-flow)