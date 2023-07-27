# Table of Contents

- [Table of Contents](#table-of-contents)
- [Optimizing Layer Performance](#optimizing-layer-performance)
- [References](#references)

<br>

# Optimizing Layer Performance

이번 챕터에서는 아래의 레이어들에 대해 최적화하는 방법을 자세히 설명하고 있다.

### Concatenation Layer

만약 implicit batch dimension을 사용한다면, 고려해야하는 것은 여러 outputs들이 함께 concat되는 경우 batch dimension에 걸쳐 브로드캐스트될 수 없으면 명시적으로 복사해야 한다는 것이다. 대부분은 레이어들은 데이터 복사를 피하기 위해 batch dimension에서 브로드캐스팅을 지원한다. 하지만 output이 다른 텐서와 concat되면 이 기능은 비활성화된다.

### Gather Layer

Gather layer에서 최대 성능을 얻으려면 `axis` of `0`을 사용해야 한다. 그리고 Gather layer에서는 사용할 수 있는 퓨전이 없다.

### Reduce Layer

Reduce layer에서 최대 성능을 얻으려면 마지막 차원에 대해서 reduction을 수행해야 한다. 순차적인 메모리 위치를 통한 read/write 패턴으로 최적의 메모리를 가능하도록 한다. 만약 일반적인 reduction operation을 수행하는 경우, 가능하다면 단일 연산으로 퓨전되는 방식으로 reduction을 표현하면 좋다.

### RNN Layer

가능하다면 `legacy RNN` 보다는 더 최신의 `RNNv2`를 사용하도록 한다. `RNNv2`는 가변 시퀀스 길이와 가변 배치 사이즈를 지원하며, 더 일관된 인터페이스를 제공한다. 최대 성능을 얻으려면 배치의 크기가 크면 클수록 좋다. 일반적으로 64의 배수일 때 가장 높은 성능을 얻는다. Bidirectional RNN-mode에서는 종속성이 추가되어 wavefront propagation을 방지하여 속도가 느려지는 경향이 있다.

또한, 새로 도입된 Loop-based API는 미리 정의된 `RNNv2` 인터페이스에 제한되지 않는 훨씬 더 유연한 메커니즘을 제공한다. `ILoopLayer` recurrence를 사용하면 loop fusion, unrolling, loop-invariant code motion 등 다양한 automatic loop optimization을 적용할 수 있다. 예를 들어, 동일한 `MatrixMultiply` 또는 `FullyConnected` 레이어의 여러 인스턴스를 결합하여 sequence dimension을 따라 loop unrolling을 적용한 후 머신의 이용률을 극대화할 때 상당한 성능 향상을 얻을 수 있다. 이는 sequence dimension을 따라 recurrent data dependence를 가진 `MatrixMultiply` 또는 `FullyConnected` 레이어를 피할 수 있는 경우에 가장 잘 동작한다 (?).

### Shuffle

Shuffle layer의 입력 텐서의 이 shuffle layer에서만 사용되고 이 레이어의 input/output 텐서의 네트워크의 input/output이 아닌 경우, underlying data에 대한 identity operations와 동일한 shuffle operations는 생략된다. TensorRT는 이러한 연산을 위한 추가 커널이나 메모리 복사를 실행하지 않는다.

### TopK

TopK layer에서 최적을 성능을 얻으려면 작은 `K` 값을 사용하여 데이터의 마지막 차원을 감소시켜 최적의 sequential memory accesses를 가능하도록 한다. Shuffle layer를 사용하여 데이터를 reshape하고 index 값을 적절히 재해석하여 한 번에 여러 차원에 대한 reduction을 시뮬레이션할 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: Optimizing Layer Performance](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-layer)
- [NVIDIA TensorRT Operatiors Documentation](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html)