# Table of Contents

- [Table of Contents](#table-of-contents)
- [CUDA Events](#cuda-events)
- [References](#references)

<br>

# CUDA Events

CUDA 런타임은 어플리케이션 내에서 비동기식으로 특정 시점에 이벤트를 레코딩하여 device의 진행사항을 모니터링하거나 정확한 타이밍을 수행하는 방법을 제공한다. 또한, 이러한 이벤트들이 완료되었는지 쿼리할 수도 있다. 특정 이벤트가 레코딩되기 이전의 모든 작업(or, 지정된 스트림의 모든 작업)들이 완료되어야 이벤트가 완료되었다고 판단한다. Stream 0(default stream)의 이벤트는 이전에 모든 스트림에서 호출된 모든 선행 작업들이 완료된 이후에 해당 이벤트가 완료된다.

CUDA 문서 내에서 언급하고 있는 CUDA Event에 대한 모든 내용은 아래 포스팅에서 자세히 다루고 있다. 아래 포스팅을 참조 바란다.

- [Introducing CUDA Events](/cuda/study/15_introducing_cuda_event.md)

<br>

# References

- [NVIDIA CUDA Documentations: CUDA Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)