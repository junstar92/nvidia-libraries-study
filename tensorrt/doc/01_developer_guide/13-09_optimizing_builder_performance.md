# Table of Contents

- [Table of Contents](#table-of-contents)
- [Optimizing Builder Performance](#optimizing-builder-performance)
  - [Timing Cache](#timing-cache)
  - [Tactic Selection Heuristic](#tactic-selection-heuristic)
- [References](#references)

<br>

# Optimizing Builder Performance

각 레이어에서 TensorRT builder는 가장 빠른 추론 엔진 플랜을 탐색하기 위해 가능한 모든 tactics를 프로파일링한다. 만약 모델이 수 많은 레이어로 구성되어 있거나 토폴로지가 복잡한 경우에는 builder time이 길어질 수 있다. 이 문서에서는 builder time을 감소시킬 수 있는 옵션들을 제공한다.

## Timing Cache

Builder time을 감소시키기 위해서 TensorRT는 layer-profiling 정보를 유지하는 layer timing cache를 생성한다. 여기에 포함된 정보는 argeted device, CUDA, TensorRT version, 그리고 `BuilderFlag::kTF32` 또는 `BuilderFlag::kREFIT`과 같이 레이어 구현을 변경시킬 수 있는 `BuilderConfig` 파라미터에 따라 다르다.

만약 동일한 IO 텐서 configuration 및 레이어 파라미터를 가진 다른 레이어가 있는 경우, TensorRT builder는 프로파일링을 스킵하고 캐시된 결과를 재사용한다. 캐시에 쿼리된 타이밍이 없다면 builder는 레이어의 시간을 측정하고 캐시를 업데이트한다.

Timing cache는 serialization과 deserialization될 수 있다. `IBuilderConfig::createTimingCache`를 사용하여 버퍼로부터 serialized cache를 로드할 수 있다.
```c++
ITimingCache* cache = config->createTimingCache(cacheFile.data(), cacheFile.size());
```

버퍼의 크기를 0으로 설정하면 새로운 빈 timing cache를 생성한다.

그런 다음 엔진을 빌드하기 전에 cache를 builder configuration에 연결해준다.
```c++
config->setTimingCache(*cache, false);
```

빌드하는 동안 cache miss 결과로 인해 timing cache에 더 많은 정보가 추가될 수 있다. 빌드 이후에는 다른 builder에서 사용하기 위해 serialization할 수 있다.
```c++
IHostMemory* serializedCache = cache->serialize();
```

만약 builder에 연결된 timing cache가 없다면 builder는 임시 local cache를 생성하여 사용하고 빌드가 끝나면 destroy한다.

Cache는 algorithm selector와 호환되지 않는다. 이때는 `BuilderFlag`를 설정하여 비활성화할 수 있다.
```c++
config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
```

## Tactic Selection Heuristic

TensorRT는 레이어 프로파일링 단계에서 builder time을 최소화하기 위해 휴리스틱 기반의 tactic selection을 허용한다. Builder는 주어진 problem size에 대한 tactic timing을 예측하고 레이어 프로파일링 단계 이전에 빠르지 않을 것 같은 tactics을 정리한다. 예측을 잘못한 경우에는 휴리스틱을 사용하지 않았을 때맡큼 성능이 좋지 않을 수 있다. 이 기능은 `BuilderFlag`를 통해 활성화할 수 있다.
```c++
config->setFlag(BuilderFlag::kENABLE_TACTIC_HEURISTIC);
```

> 이 기능은 NVIDIA Ampere 아키텍처 이상의 GPU에서만 지원된다.

<br>

# References

- [NVIDIA TensorRT Documentation: Optimizing Builder Performance](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt-builder-perf)