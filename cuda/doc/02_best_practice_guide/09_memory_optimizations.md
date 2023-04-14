# Table of Contents

- [Table of Contents](#table-of-contents)
- [Memory Optimizations](#memory-optimizations)
- [Data Transfer Between Host and Device](#data-transfer-between-host-and-device)
- [Device Memory Spaces](#device-memory-spaces)
- [Allocation](#allocation)
- [NUMA Best Practices](#numa-best-practices)
- [References](#references)

<br>

# Memory Optimizations

메모리 최적화는 성능에서 가장 중요한 영역이다. 메모뢰 최적화의 목표는 bandwidth를 극대화하여 하드웨어의 최대한 사용하는 것이다. Bandwidth 측면에서는 가능한 빠른 메모리를 많이 사용하고 액세스가 느린 메모리를 덜 사용하는 것이 가장 좋다. Memory Optimizations 챕터에서는 host와 device에서의 다양한 종류의 메모리에 대해 논의하고 어떻게 이들을 효율적으로 사용하는지에 대해서 다룬다.

<br>

# Data Transfer Between Host and Device

[Data Transfer Between Host and Device](/cuda/doc/02_best_practice_guide/09-01_data_transfer_between_host_and_device.md) 참조

<br>

# Device Memory Spaces

[Device Memory Spaces](/cuda/doc/02_best_practice_guide/09-02_device_memory_spaces.md) 참조

<br>

# Allocation

`cudaMalloc()`과 `cudaFree()`를 통한 device memory 할당 및 해제는 비용이 꽤 큰 operation이다. Device memory를 관리할 때는 stream ordered pool allocator인 `cudaMallocAsync()`와 `cudaFreeAsync()`를 사용하는 것이 좋다.

<br>

# NUMA Best Practices

최근 몇몇 리눅스 배포판에서는 기본적으로 automatic NUMA balancing(or [AutoNUMA](https://lwn.net/Articles/488709/))를 활성화한다. 경우에 따라 NUMA balancing에 의해 NVIDIA GPU에서 실행되는 프로그램의 성능이 저하될 수 있다. 최적의 성능을 위해서는 사용자가 프로그램의 NUMA 특성을 수동으로 조정해야 한다.

최적의 NUMA 조정은 각 어플리케이션 및 노드의 특성, 원하는 하드웨어 친화성 정도에 따라 다르지만 일반적으로는 automatic NUMA balancing을 비활성화하는 것이 좋다.

<br>

# References

- [NVIDIA CUDA Documentation: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)