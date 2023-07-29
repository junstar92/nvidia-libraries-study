# Table of Contents

- [Table of Contents](#table-of-contents)
- [Algorithm Selection and Reproducible Builds](#algorithm-selection-and-reproducible-builds)
- [References](#references)

<br>

# Algorithm Selection and Reproducible Builds

TensorRT optimizer는 기본적으로 엔진의 전체적인 실행 시간을 최소화하는 알고리즘을 선택한다. 때로는 각 구현의 실행 시간들이 비슷한 경우에는 시스템의 노이즈에 의해 특정 구현이 선택될 수 있다. 서로 다른 구현에서는 일반적으로 부동소수점 값의 누적 순서가 다르고, 두 구현이 서로 다른 알고리즘을 사용하거나 다른 정밀도로 실행될 수 있다. 따라서 builder를 호출할 때마다 일반적으로 비트 수준으로 동일한 결과를 반환하는 엔진이 생성되지 않는다.

하지만 deterministic build가 필요하거나 이전 빌드에서 선택된 알고리즘으로 엔진을 다시 생성하는 것이 중요할 때가 있다. 이런 경우에 `IAlgorithmSelector` 인터페이스의 구현을 제공하고 `setAlgorithmSelector`를 사용하여 builder configuration에 이를 연결하면 알고리즘 선택을 수동으로 가이드할 수 있다.

`IAlgorithmSelector::selectAlgorithms` 메소드는 레이어의 알고리즘 요구 사항에 대한 정보를 가지고 있고 이러한 요구 사항을 충족하는 `Algorithm` 선택 집합이 포함된 `AlgorithmContext`를 받는다. 이 메소드는 TensorRT가 이 레이어에 대해 고려해야 할 알고리즘 집합을 반환한다.

Builder는 이러한 알고리즘으로부터 네트워크의 global runtime을 최소화하는 알고리즘을 선택한다. 만약 어떠한 알고리즘도 선택되지 않으면서 `BuilderFlag::kREJECT_EMPTY_ALGORITHMS`가 설정되지 않았다면, TensorRT는 이것을 이 레이어가 모든 알고리즘을 사용할 수 있다고 해석한다. 만약 empty list가 반환되는 경우에 에러를 발생시키려면 `BuilderFlag::kREJECT_EMPTY_ALGORITHMS` 플래그를 설정하면 된다.

TensorRT가 주어진 프로파일에 대한 네트워크 최적화를 완료한 후, `IAlgorithmSelector`는 `reportAlgorithms`를 호출한다. 이 메소드는 각 레이어에서 최종 선택된 알고리즘을 리포트하는데 사용된다.

결정적인 TensorRT 엔진을 빌드하기 위해서는 `selectAlgorithms`로부터 하나의 choice만 리턴하면 된다. 이전 빌드로부터 선택된 알고리즘을 리플레이하려면 `reportAlgorithms`를 사용하여 해당 빌드에서 선택된 알고리즘을 기록하고 이를 `selectAlgorithms`에서 반환한다.

[sampleAlgorithmSelect](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleAlgorithmSelector)에서 algorithm selector를 사용하여 builder에서 determinisim과 reproducibility를 당성하는 방법을 자세히 설명한다.

> **Note:**
> - Algorithm selection에서의 `layer`는 `INetworkDefinition`에서의 `ILayer`와는 다르다. 여기서의 `layer`는 fusion optimization으로 인한 여러 network layers의 집합과 동등할 수 있다.
> - `selectAlgorithm`에서 가장 빠른 알고리즘을 선택해도 전체 네트워크에서 최적의 성능을 달성하지 못할 수 있다 (reformatting overhead로 인해).
> - 만약 TensorRT가 발견한 레이어가 `no-op`라면 `selectAlgorithm`에서 `IAlgorithm`의 timing은 0이다.
> - `reportAlgorithms`는 `selectAlgorithms`에서 제공되는 `IAlgorithm`에 대한 timing과 workspace 정보를 제공하지 않는다.

<br>

# References

- [NVIDIA TensorRT Documentation: Algorithm Selection and Reproducible Builds](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select)
- [NVIDIA TensorRT Samples: sampleAlgorithmSelector](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleAlgorithmSelector)