# Table of Contents

- [Table of Contents](#table-of-contents)
- [Builder Optimization Level](#builder-optimization-level)
- [References](#references)

<br>

# Builder Optimization Level

Builder config에 optimization level을 설정하여 TensorRT가 잠재적으로 더 좋은 성능의 tactics를 찾는데 소모하는 시간을 조정할 수 있다. 기본 optimization level은 3이다. 더 작은 값으로 설정하면 엔진을 빌드하는 시간은 훨씬 빨라지지만 엔진의 성능이 저하될 수 있다. 반면 더 큰 값으로 설정하면 엔진 빌드 시간은 늘어나지만 TensorRT가 성능이 더 좋은 tactics을 찾을 수 있고 이로 인해 성능이 좋아질 수 있다.

예를 들어, 엔진 빌드 시간이 가장 빠르도록 optimization level을 0으로 설정하려면 다음과 같이 설정해주면 된다.

```c++
config->setOptimizationLevel(0);
```

<br>

# References

- [NVIDIA TensorRT Documentation: Builder Optimization Level](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt-builder-optimization-level)