# Table of Contents

- [Table of Contents](#table-of-contents)
- [Page-Locked Host Memory](#page-locked-host-memory)
  - [Portable Memory](#portable-memory)
  - [Write-Combining Memory](#write-combining-memory)
  - [Mapped Memory](#mapped-memory)
- [References](#references)

<br>

# Page-Locked Host Memory

CUDA Runtime API에서는 page-locked(=pinned) host memory를 사용할 수 있는 함수들을 제공한다. 이와 반대되는 메모리는 regular pageable host memory이며, 이는 `malloc()`을 통해 할당할 수 있다.

- `custHostAlloc()`과 `cudaFreeHost()`를 사용하여 page-locked host memory를 할당하고 해제할 수 있다.
- `cudaHostRegister()`를 통해 `malloc()`으로 할당된 메모리를 page-lock시킬 수 있다 (limitation 존재, [API 문서](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY) 참조).

Page-locked host memory를 사용하면 아래의 이점들을 얻을 수 있다.

1. Page-locked host memory와 device memory 간 데이터 복사는 kernel execution과 동시에 수행될 수 있다. 이에 대한 내용은 [Asynchronous Concurrent Execution](/cuda/doc/01_programming_guide/03-02-08_asynchronous_concurrent_execution.md)을 참조 바람.
2. 몇몇 device에서는 page-locked memory가 device address space에 매핑될 수 있다. 이를 통해 host와 device 간의 명시적인 copy를 제거할 수 있다 ([Mapped Memory](#mapped-memory) 참조).
3. Front-side bus가 있는 시스템에서는 host memory가 page-locked로 할당되면 host memory와 device memory 간의 bandwidth가 높고, write-combining으로 할당되면 bandwidth가 더 높다고 한다 (무엇을 말하고자 하는지 파악 못함.. ㅠ).

> 단순히 page-locked memory는 `cudaMallocHost()`를 통해 할당할 수도 있다. Pinned memory 및 zero-copy memory에 대한 내용은 [Memory Management](/cuda/study/10_memory_management.md)에서 자세히 다루고 있다.

## Portable Memory

시스템에 GPU device가 여러 개인 경우, (동일한 unified address space를 공유하는) device에서 page-locked memory block을 함께 사용할 수 있다. 모든 device에서 이 메모리를 사용하려면 `cudaHostAllocPortable` 플래그를 `cudaHostAlloc()`에 전달하여 블록을 할당하거나, `malloc()`으로 할당된 메모리인 경우에는 `cudaHostRegisterPortable` 플래그를 `cudaHostRegister()`에 전달하여 page-lock 시켜주어야 한다.

## Write-Combining Memory

공식 문서의 [Write-Combining Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#write-combining-memory) 참조

> 하드웨어와 엮인 부분인 것 같은데, 여기에 대한 지식이 별로 없어서 완전히 이해를 하지 못했다. 특수한 상황에서 사용되는 것 같다.

## Mapped Memory

> [vector_add_zerocopy.cu](/cuda/code/vector_add/vector_add_zerocopy.cu)에서 mapped page-locked memory를 사용하는 방법을 보여준다.

Page-locked host memory의 블록은 `cudaHostAlloc()`에 `cudaHostAllocMapped` 플래그를 전달하거나 `malloc()`으로 할당된 메모리의 경우에는 `cudaHostRegisterMapped` 플래그를 `cudaHostRegister()`에 전달하여 device의 address space에 매핑될 수 있다. 이 메모리 블록은 일반적으로 두 개의 주소를 갖는데, 하나는 `cudaHostAlloc()` 또는 `malloc()`으로부터 반환받은 host memory의 주소이고 다른 하나는 `cudaHostGetDevicePointer()`를 사용하여 얻은 device memory의 주소이다. 여기서 device memory의 주소를 사용하면 커널 내에서 이 블록에 액세스할 수 있다.

커널 내에서 host memory에 직접 액세스할 때는 device memory와 같은 bandwidth가 아니며, 물리적으로 더 멀기 때문에 느릴 수 있다. 하지만, 다음의 몇 가지 이점들이 존재한다.

- Host memory에서 device memory로의 데이터 복사가 필요없으며, 데이터의 전달은 커널이 필요할 때 암시적으로 수행된다.
- Kernel execution과 data transfer를 오버랩하기 위해 스트림을 사용할 필요가 없다 (kernel-originated data는 자동으로 kernel execution과 오버랩된다).

Mapped page-locked memory는 host와 device간에 서로 공유되므로, 프로그램 내에서 잠재적인 read-after-write, write-after-read, 또는, write-after-write 위험을 피하기 위해 스트림이나 이벤트를 사용하여 메모리 액세스를 동기화시켜 주어야 한다.

> 문서에서는 mapped page-locked memory에 대한 device pointer를 검색하려면 `cudaSetDeviceFlags()`를 `cudaDeviceMapHost` 플래그를 지정하여 호출해야 한다고 언급하고 있다. 그렇지 않으면 `cudaHostGetDevicePointer()` 호출은 에러를 리턴한다고 한다. 하지만, [vector_add_zerocopy.cu](/cuda/code/vector_add/vector_add_zerocopy.cu)에서는 `cudaSetDeviceFlags()`를 호출하지 않아도 `cudaHostGetDevicePointer()`가 잘 동작한다.

> `cudaHostGetDevicePointer()`는 해당 device가 mapped page-locked host memory를 지원하지 않는 경우에도 에러를 반환한다. 프로그램 내에서는 device query를 통해 `cudaDeviceProp::canMapHostMemory` 멤버 변수를 확인하여 지원하는지 체크할 수 있다.

> Mapped page-locked memory에 대한 atomic function은 host나 다른 device 관점에서는 atomic이 아니다.

<br>

# References

- [NVIDIA CUDA Documentations: Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
- CUDA Sample Code: [simpleZeroCopy](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleZeroCopy)