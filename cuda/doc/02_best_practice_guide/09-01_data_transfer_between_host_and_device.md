# Table of Contents

- [Table of Contents](#table-of-contents)
- [Data Transfer Between Host and Device](#data-transfer-between-host-and-device)
  - [Pinned Memory](#pinned-memory)
  - [Asynchronous and Overlapping Trasfers with Computation](#asynchronous-and-overlapping-trasfers-with-computation)
  - [Zero Copy](#zero-copy)
  - [Unified Virtual Addressing](#unified-virtual-addressing)
- [References](#references)

<br>

# Data Transfer Between Host and Device

Device memory와 GPU 간의 peak theoretical bandwidth(Tesal V100의 경우, 898GB/s)는 host memory와 device memory 간의 peak theoretical bandwidth(16GB/s on PCIe x16 Gen3)보다 훨씬 높다. 따라서, 최상의 성능을 위해서는 host와 device 간의 데이터 전송을 최소화하는 것이 중요하다.

> Host CPU에서 실행하는 것과 비교했을 때, GPU에서 성능 향상이 그리 크지 않다면 host와 device 간의 데이터 전송을 최소화하고 그냥 CPU에서 실행하는 것이 좋을 수 있다.

또한, 데이터 전송과 관련한 오버헤드가 꽤 크기 때문에, 작은 크기의 데이터 전송을 많이 수행하는 것보다 하나의 큰 크기의 데이터 전송으로 일괄 처리하는 것이 훨씬 더 좋은 성능을 발휘한다. 이렇게 하려면 메모리의 비연속적인 영역을 연속적인 버퍼에 패킹하고, 데이터를 전송한 이후에 다시 언패킹해야 한다.

마지막으로 page-locked(pinned) memory를 사용하면 host와 device 간에 높은 bandwidth를 달성할 수 있다. Pinned memory에 관한 내용은 바로 아래에서 다룬다.

## Pinned Memory

Page-locked(or pinned) memory를 사용하면 host와 device 사이에서 가장 높은 bandwidth를 얻을 수 있다. 예를 들어, PCIe x16 Gen3 카드에서 pinned memory는 대략 12GB/s의 전송 속도를 갖는다.

Pinned memory는 런타임 API `cudaHostAlloc()`을 통해 할당할 수 있다. CUDA Sample의 [bandwidthTest](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest)를 통해 pageable memory와 pinned memory 간의 메모리 전송 성능을 측정 및 비교할 수 있다. 이 코드를 컴파일하고 실행하면 아래와 같은 출력 결과를 확인할 수 있다. Pageable memory와 pinned memory에 대한 HtoD, DtoH, DtoD 메모리 복사 성능을 측정한다.
```
$ ./bandwidthTest --memory=pageable
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 3080
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     15.9

 Device to Host Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     11.5

 Device to Device Bandwidth, 1 Device(s)
 PAGEABLE Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     649.8

Result = PASS

$ ./bandwidthTest --memory=pinned
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 3080
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.2

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     27.0

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     651.9

Result = PASS
```
위 결과는 pinned memory의 DtoH, HtoD memory transfer가 pageable memory보다 빠르다는 것을 보여준다. DtoD 메모리 복사에서는 두 메모리가 큰 차이가 없다는 것을 알 수 있다.

이미 할당된 시스템 메모리 영역의 경우에는 `cudaHostRegister()`를 사용하여 별도의 버퍼를 할당하고 데이터를 복사할 필요없이 바로 pinned memory로 만들 수 있다.

> Pinned memory에 관한 내용은 아래 포스팅을 참조 바람
> - [Page-Locked Host Memory](/cuda/doc/01_programming_guide/03-02-06_page_locked_host_memory.md)
> - [Memory Management: Pinned Memory](/cuda/study/10_memory_management.md#pinned-memory)

> Pinned memory를 너무 많이 사용하면 시스템 성능이 저하될 수 있다. 또한, 메모리를 고정시키는 것을 일반 메모리 할당보다 오버헤드가 크다.

## Asynchronous and Overlapping Trasfers with Computation

`cudaMemcpy()`를 사용한 host와 device간 메모리 전송은 blocking이다. 즉, 데이터 전송이 끝나야만 host thread로 제어권이 반환된다. 반면, `cudaMemcpyAsync()`는 `cudaMemcpy()`의 non-blocking 버전이며 host thread로 제어권을 즉시 반환한다. `cudaMemcpy()`와는 달리, `cudaMemcpyAsync()`는 pinned host memory를 사용해야 하며 스트림 ID를 추가 인자로 받는다. 스트림은 단순히 device에서 순서대로 수행되는 일련의 작업이며, 서로 다른 스트림의 작업은 동시에 실행될 수 있다. 따라서, 서로 다른 스트림의 작업이 서로 오버랩될 수 있으며, 이를 사용하여 host와 device 간의 데이터 전송과 연산을 오버랩할 수 있는 것이다.

비동기 전송은 데이터 전송과 연산을 두 가지 다른 방식으로 오버랩할 수 있다. 모든 CUDA-enabled device에서 host 연산을 비동기 데이터 전송과 device 연산을 오버랩할 수 있다. 예를 들어, 아래 코드에서는 device에서 데이터 전송과 커널 실행이 수행되는 동안 host 연산(`cpuFunction()`)이 수행된다.
```c++
// Overlapping cpu computation and data transfers
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
kernel<<<grid, block>>>(a_d);
cpuFunction();
```
`cudaMempcyAsync()`의 마지막 인자는 스트림 ID이며, 위 코드에서는 default 스트림(stream 0)이 전달되었다. 커널 또한 execution configuration으로 스트림 ID가 지정되지 않았기 때문에 default 스트림에서 실행되는데, 커널은 데이터 전송이 완료된 이후에 실행된다. 따라서, 데이터 전송과 커널 실행 간의 명시적인 동기화가 필요없다. 위 코드에서 메모리 복사와 커널은 호출 즉시 제어권을 host로 반환하므로, 이들과 host 함수 `cpuFunction()`은 서로 오버랩된다.

방금 예제 코드에서는 메모리 복사와 커널 실행이 순차적으로 수행된다. **Concurrent copy and compute**를 지원하는 device에서는 kernel execution과 host와 device 간의 data transfer를 오버랩할 수 있다. 물론 두 연산은 오버랩하려면 `pinned host memory`를 사용해야 하며, 데이터 전송과 커널은 서로 다른 스트림(default 스트림이 아닌)을 사용해야 한다.

> Concurrent copy and compute 지원 여부는 device 정보를 쿼리하여 `cudaDeviceProp` 구조체의 `asyncEngineCount` 필드를 통해 확인할 수 있다.

아래 코드는 **conccurent copy and execute**를 보여준다.
```c++
// Concurrent copy and execute
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(otherData_d);
```
여기서는 두 개의 스트림을 생성하고, 각각 데이터 전송과 커널 실행에 사용한다.

Concurrent copy and execute는 kernel execution과 asynchronous data trasnfer가 어떻게 오버랩되는지를 잘 보여준다. 이 기법은 큰 데이터를 여러 개의 chunk로 나누고, 각 chunk 단위로 데이터 전송-커널 실행이 가능한 경우에 사용할 수 있다 (데이터간 종속성이 없는 경우). 아래의 코드들은 sequential copy and execute와 staged concurrent copy and execute를 각각 보여준다. 이들의 결과는 동일하다. 
```c++
// Sequential copy and execute
cudaMemcpy(a_d, a_h, N*sizeof(float), dir);
kernel<<<N/nThreads, nThreads>>>(a_d);
```
```c++
// Staged concurrent copy and execute
size=N*sizeof(float)/nStreams;
for (i=0; i<nStreams; i++) {
    offset = i*N/nStreams;
    cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
    kernel<<<N/(nThreads*nStreams), nThreads, 0,
             stream[i]>>>(a_d+offset);
}
```
Staged concurrent copy and execute은 데이터 전송과 커널 실행을 `nStreams` 단계로 나누어서 수행하며, 데이터는 `nThreads * nStreams`로 균등하게 나눌 수 있다고 가정한다. 

스트림 내에서는 순차적으로 실행되기 때문에 각 스트림에서의 데이터 전송이 완료되기 전까지는 어떠한 커널도 실행되지 않는다. GPU는 비동기 데이터 전송과 커널 실행을 동시에 처리할 수 있다. 하나의 copy engine을 가진 GPU는 하나의 비동기 데이터 전송과 커널을 동시에 수행할 수 있지만, 두 개의 copy engines를 가진 GPU는 비동기 HtoD 데이터 전송, 비동기 DtoH 데이터 전송과 커널 실행을 동시에 수행할 수 있다. 아래 그림은 Sequential copy and execute와 Staged concurrent copy and execute의 타임라인을 보여준다 (위: Sequential / 아래: Concurrent).

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/timeline-comparison-for-copy-and-kernel-execution.png" width=400px style="display: block; margin: 0 auto; background-color:white"/>

> 이와 관련한 내용을 아래 포스팅에서 다룬다.
> - [Asynchronous Concurrent Execution](/cuda/doc/01_programming_guide/03-02-08_asynchronous_concurrent_execution.md)
> - [Introducing CUDA Streams](/cuda/)

## Zero Copy

Zero copy는 CUDA 2.2에서 추가된 기능이며, 이를 사용하면 GPU 스레드가 host memory에 직접 액세스할 수 있다. 이를 사용하려면 **mapped pinned memory**(non-pageable)를 필요로 한다. **Integrated GPU**에서는 integrated GPU와 CPU memory가 물리적으로 동일하기 때문에 불필요한 복사를 피하므로 항상 성능이 향상된다. **Discreted GPU**에서는 특정 경우에서만 유리하다. Discreted GPU에서는 데이터가 GPU에 캐싱되지 않기 때문에 mapped pinned memory는 오직 한 번만 읽거나 써야하며 메모리 액세스는 병합되어야 한다. Zero copy를 사용하면 명시적인 데이터 전송이 없고, 커널과 자동으로 오버랩되기 때문에 스트림 대신 사용할 수 있다.

Zero copy를 사용하는 일반적인 방법은 다음과 같다.
```c++
// Zero-copy host code
float *a_h, *a_map;
...
cudaGetDeviceProperties(&prop, 0);
if (!prop.canMapHostMemory)
    exit(0);
cudaSetDeviceFlags(cudaDeviceMapHost);
cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&a_map, a_h, 0);
kernel<<<gridSize, blockSize>>>(a_map);
```

> Mapped pinned memory를 사용하면 CUDA 스트림을 사용하지 않고 연산과 CPU-GPU 메모리 전송을 오버랩할 수 있다. 하지만, 이 메모리 영역에 반복적으로 액세스하면 CPU-GPU 메모리 전송이 반복적으로 수행되므로, 이전에 읽은 데이터를 device 메모리에 직접 캐싱하는 것이 좋다.

## Unified Virtual Addressing

Compute capability 2.0 이상의 device부터 64-bit 리눅스와 윈도우에서 **Unified Virtual Addressing**(UVA)라는 특별한 addressing mode를 지원한다. 이를 사용하면 모든 device에서 host memory와 device memory가 단일 가상 주소 공간(a single virtual address space)를 공유한다.

UVA가 지원되기 전에는 프로그램 내에서 어떤 포인터가 device memory를 참조하고, host memory를 참조하는지 별도의 메타데이터 비트로 추적해야 했다. 반면 UVA를 사용하면 `cudaPointerGetAttributes()`를 사용하여 포인터의 값을 검사하여 포인터가 가리키는 물리적 메모리 공간을 간단하게 알 수 있다.

UVA에서 `cudaHostAlloc()`으로 할당된 pinned host memory는 동일한 host/device memory 포인터를 가지므로 `cudaHostGetDevicePointer()`를 사용할 필요가 없다. 그러나 이미 시스템에 할당된 host memory를 `cudaHostRegister()`로 고정시키는 경우에는 host pionter와 다른 device pointer를 가지므로 `cudaHostGetDevicePointer()`를 필요로 한다.

> UVA는 PCIe bus 또는 NVLink를 통해 GPU간 peer-to-peer(P2P) data transfer를 활성화하는데 필요하다.

<br>

# References

- [NVIDIA CUDA Documentation: Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)
- [CUDA Samples: bandwidthTest](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest)