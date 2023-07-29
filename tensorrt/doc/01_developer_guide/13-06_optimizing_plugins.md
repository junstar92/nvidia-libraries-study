# Table of Contents

- [Table of Contents](#table-of-contents)
- [Optimizing Plugins](#optimizing-plugins)
- [References](#references)

<br>

# Optimizing Plugins

TensorRT에서는 레이어 연산을 수행하는 커스텀 플로그인 등록 메커니즘을 제공한다. Plugin creator를 레지스트리에 등록한 후, 사용자는 레지스트리를 조희하여 해당 plugin creator를 찾고, plugin 객체를 serialization/deserialization 중에 네트워크에 추가할 수 있다.

모든 TensorRT (built-in)플러그인은 플러그인 라이브러리가 로드되면 자동으로 등록된다. 커스텀 플러그인에 대한 내용은 아래 링크에서 자세히 다루고 있다.

- [Extending TensorRT with Custom Layers](/tensorrt/doc/01_developer_guide/09_extending_tensorrt_with_custom_layers.md)
- [Plugin Example](/tensorrt/study/06_plugin.md)

플러그인의 성능은 해당 플러그인에서 작업을 수행하는 CUDA 코드에 따라 다르다. 플러그인을 개발할 때는 플러그인에서 수행하는 작업을 수행하고 정확성을 검증하는 간단한 CUDA 어플리케이션으로부터 시작하는 것이 도움이 될 수 있다. 이 프로그램은 성능 측정, 단위 테스트 및 대체 구현으로 확장될 수 있다. 그리고 최종적으로 플러그인으로 TensorRT로 통합할 수 있다.

최상의 성능을 얻으려면 플러그인에서 가능한 많은 타입을 지원하는 것이 중요하다. 그러면 네트워크를 실행하는 동안 reformatting 작업이 필요하지 않다.


<br>

# References

- [NVIDIA TensorRT Documentation: Optimizing Plugins](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-plugins)