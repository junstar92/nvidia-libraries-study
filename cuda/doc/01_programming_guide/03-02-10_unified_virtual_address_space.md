# Table of Contents

- [Table of Contents](#table-of-contents)
- [Unified Virtual Address Space](#unified-virtual-address-space)
- [References](#references)

<br>

# Unified Virtual Address Space

64비트 프로세스로 어플리케이션이 실행될 때, host와 모든 device(compute capability 2.0 이상)는 하나의 address space를 사용할 수 있다. CUDA API 호출을 통해 생성된 모든 host memory allocation과 모든 device memory allocation은 이 virtual address 범위 내에 있다.

이를 통해,

- CUDA를 통해 host에서 할당된 모든 메모리 위치, 또는 devices에서 할당된 모든 메모리의 위치는 **unified address space**를 사용하며, 이 주소는 `cudaPointerGetAttributes()`를 사용하여 포인터 값으로 확인할 수 있다.
- Unified address space를 사용하는 모든 device의 메모리로 또는 메모리로부터 복사할 때, `cudaMemcpy*()`의 `cudaMemcpyKind` 파라미터를 `cudaMemcpyDefault`로 설정하여 포인터로부터의 위치를 결정할 수 있다. 사용 중인 GPU device가 unified addressing을 사용한다면, CUDA를 통해 할당되지 않는 host 포인터에 대해서도 동작한다.
- `cudaHostAlloc()`을 통한 할당은 unified address space를 사용하는 모든 devices에 대해서 자동으로 **portable** 하다. 즉, 장치 간 이동이 가능하다. `cudaHostAlloc()`으로부터 반환된 포인터는 이들 devices에서 실행되는 커널 내에서 직접 사용할 수도 있다 (따라서, `cudaHostGetDevicePOinter()` API를 통해 device pointer를 쿼리할 필요가 없다).

> [Memory Management](/cuda/study/10_memory_management.md#unified-virtual-addressing-uva)에서 unified virtual address에 대해 다루고 있다.

> 만약 특정 device에서 unified address를 사용할 수 있는지 확인하고 싶다면, `cudaGetDeviceProperties()`를 통해 쿼리한 device property에서 `cudaDeviceProp::unifiedAddressing` 멤버 변수를 확인하면 된다.

<br>

# References

- [NVIDIA CUDA Documentations: Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space)