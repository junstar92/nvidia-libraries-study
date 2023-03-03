# Table of Contents

- [Table of Contents](#table-of-contents)
- [Memory Management](#memory-management)
  - [Memory Allocation and Deallocation](#memory-allocation-and-deallocation)
  - [Memory Transfer](#memory-transfer)
- [Pinned Memory](#pinned-memory)
- [Zero-Copy Memory](#zero-copy-memory)
- [Unified Virtual Addressing (UVA)](#unified-virtual-addressing-uva)
- [Unified Memory](#unified-memory)
- [References](#references)

<br>

# Memory Management

CUDA 프로그래밍에서의 메모리 관리는 C 프로그래밍과 유사하다. 여기에 추가로 host와 device 간의 메모리 이동이 추가된다. NVIDIA에서는 [**Unifed Memory**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming)를 도입해서 host와 device memory를 통합하고 있지만 아직 대부분의 프로그램에서는 명시적인 데이터 이동을 사용하는 것으로 보인다. 이번 포스팅에서는 CUDA runtime API를 통해 명시적으로 데이터를 관리하고 이동시키는 방법에 대해서 살펴본다.

- Allocate and deallocate device memory
- Transfer data between the host and device

<br>

## Memory Allocation and Deallocation

CUDA 프로그래밍 모델에서는 heterogeneous system은 host와 device로 구성되어 있으며, 각각은 서로 분리된 메모리 공간을 가지고 있다고 가정한다. CUDA Kernel 함수는 device memory 공간에서 동작하며, CUDA runtime은 device memory를 할당하고 해제하는 함수를 제공한다. Host 측에서 global memory를 할당하려면 아래의 함수를 사용하면 된다.

```c++
cudaError_t cudaMalloc(void **devPtr, size_t count);
```

이 함수는 `count` bytes 만큼의 global memory를 할당하고, 할당된 메모리의 주소를 `devPtr`에 리턴한다. 어떤 변수 타입이든 할당된 메모리는 적절하게 정렬(aligned)되어 있다.

`cudaMalloc()`을 통해 할당된 global memory를 자동으로 그 값이 클리어되지 않기 때문에 특정 값으로 채우려면 `cudaMemset()`을 통해 초기화해주어야 한다.

```c++
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
```

할당된 global memory가 더이상 사용되지 않으면, 아래 API를 사용하여 할당된 메모리를 해제할 수 있다.

```c++
cudaError_t cudaFree(void *devPtr);
```

Device memory를 할당하고 해제하는 것은 cost가 상당한 연산이다. 그렇기 때문에 성능에 영향을 미치지 않도록 최대한 할당된 메모리를 재사용하는 것이 좋다.

<br>

## Memory Transfer

Global memory를 할당한 뒤, 다음의 함수를 사용하여 데이터를 host에서 device로 전달할 수 있다.

```c++
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```

이 함수는 `src` 메모리 공간으로부터 `count` bytes만큼을 `dst` 메모리 공간으로 복사한다. `kind`는 복사 방향을 나타내며, 아래의 값들이 가능하다.

- `cudaMemcpyHostToHost` : Host -> Host
- `cudaMemcpyHostToDevice` : Host -> Device
- `cudaMemcpyDeviceToHost` : Device -> Host
- `cudaMemcpyDeviceToDevice` : Device -> Device
- `cudaMemcpyDefault` : Direction of the transfer is inferred 

만약 `dst`와 `src` 포인터가 `kind`로 지정된 복사 방향과 일치하지 않는다면, `cudaMemcpy`에서는 undefined behavior이 발생한다. 이 함수는 대부분의 경우 host에 동기화된다.

아래 예제 코드는 `cudaMemcpy`를 사용하여 host와 device 간의 데이터 이동을 보여준다.

```c++
// memTransfer.cu
#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // memory size
    int size = 1 << 22;
    size_t bytes = size * sizeof(float);

    // get device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("> device %d: %s  memory size: %d  bytes: %5.2f MB\n", 
        dev, prop.name, size, bytes/(1024.f*1024.f));
    
    // allocate the host memory
    float* h_a = (float*)malloc(bytes);

    // allocate the device memory
    float* d_a;
    cudaMalloc(&d_a, bytes);

    // initialize the host memory
    for (int i = 0; i < size; i++) h_a[i] = 0.5f;

    // transfer data from the host to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // transfer data from the device to the host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    free(h_a);

    // reset device
    cudaDeviceReset();

    return 0;
}
```

위 코드를 작성하고 아래 커맨드를 통해 컴파일 및 `nsight system`으로 컴파일한 실행 파일을 프로파일링하게 되면,

```
$ nvcc -O3 memTransfer.cu -o memTransfer
$ nsys profile --stats=true ./memTransfer
```

아래와 같은 출력 결과를 볼 수 있다.

```
...
[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ------------------
     58.4        1,510,638      1  1,510,638.0  1,510,638.0  1,510,638  1,510,638          0.0  [CUDA memcpy DtoH]
     41.6        1,074,411      1  1,074,411.0  1,074,411.0  1,074,411  1,074,411          0.0  [CUDA memcpy HtoD]
...
```

여기서 memcpy 연산에 걸린 시간, 크기를 알 수 있으며 host -> device는 `HtoD`, device -> host는 `DtoH`로 표시된다.

<br>

# Pinned Memory

할당된 host memory는 기본적으로 **pageable** 이다. 즉, OS에 의해서 가상 메모리에 위치할 수 있고 이 메모리가 필요할 때 page fault가 발생할 수 있다는 것을 의미한다.

GPU는 host의 OS가 host memory를 실제 물리 메모리에 언제 가지고 오는지를 제어할 수 없기 때문에 pageable host memory에 안전하게 액세스할 수 없다. 따라서, pageable host memory로부터 device memory로 데이터를 전달할 때, CUDA driver는 먼저 temporary **page-locked** 또는 **pinned** host memory를 할당하고, source host data를 pinned memory로 복사한 뒤, pinned memory로부터 device memory로 전달한다. 이 과정을 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcH9Oro%2FbtrZKoOQIAf%2FffXtOkkOkPOlCwVKj8VALK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

아래의 CUDA runtime API를 사용하면 직접 pinned host memory를 할당할 수 있다.

```c++
cudaError_t cudaMallocHost(void **devPtr, size_t count);
```

이 함수는 `count` bytes 만큼의 pinned memory(page-lock host memory)를 할당한다. Pinned memory는 device에서 직접적으로 액세스할 수 있으므로 pageable memory보다 더 높은 bandwidth로 read/write가 가능하다. 단, pinned memory를 너무 많이 사용하면 host system의 성능에 영향을 미칠 수 있으므로 주의해야 한다 (사용할 수 있는 pageable memory의 크기가 줄어든다).

위 API를 통해 할당된 pinned memory는 다음의 API를 통해 해제할 수 있다.
```c++
cudaError_t cudaFreeHost(void *ptr);
```

위의 예제 코드(memTransfer.cu)에서 `malloc`을 사용하는 부분을 `cudaMallocHost`로 바꾸고, `free`를 `cudaFreeHost`로 변경하여 동일한 커맨드로 빌드 후, `nsight system`으로 프로파일링해보면 아래와 같은 출력을 확인할 수 있다.

```
...
[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ------------------
     63.0        1,055,492      1  1,055,492.0  1,055,492.0  1,055,492  1,055,492          0.0  [CUDA memcpy HtoD]
     37.0          620,962      1    620,962.0    620,962.0    620,962    620,962          0.0  [CUDA memcpy DtoH]
...
```

매우 큰 차이라고 볼 수는 없지만, 기존보다 더 빨라진 것을 확인할 수 있다.

> Pinned memory 할당 비용은 pageable memory보다 크지만, 대용량 메모리 전송에서 더 높은 처리량을 제공한다. Pageable memory에 비해 pinned memory를 사용할 때 달성할 수 있는 속도 향상은 compute capability에 따라 다르다.
>
> Host와 device 간의 메모리 전송은 커널 실행과 오버랩될 수 있는데, 이는 스트림을 사용하여 달성할 수 있다. 최적의 성능을 위해서는 메모리 전송을 최소화하거나 커널 실행과 오버랩이 되도록 설계해야 한다.

<br>

# Zero-Copy Memory

일반적으로 host는 device 변수에 직접 액세스할 수 없고, device 또한 host 변수에 직접 액세스할 수 없다. 여기에는 한 가지 예외가 있는데, 이것이 바로 **zero-copy memory** 이다. Host와 device는 모두 zero-copy memory에 액세스할 수 있다.

GPU 스레드들은 zero-copy memory에 직접 액세스할 수 있는데, 이를 사용할 때 얻을 수 있는 몇 가지 이점들은 다음과 같다.

- device memory가 충분하지 않을 때 host memory를 활용할 수 있음
- host <-> device 간의 명시적인 데이터 전달이 필요없음
- PCIe transfer rates 향상

Zero-copy memory를 사용하여 host와 device에서 데이터를 공유할 때는 반드시 host와 device에서의 액세스 간의 동기화가 필요하다. 만약 host와 device 모두에서 zero-copy memory를 동시에 수정하려고하면 undefined behavior이 발생한다.

Zero-copy memory는 device address space로 매핑되는 pinned memory이다. `cudaHostAlloc` API를 사용하면 mapped, pinned memory를 할당할 수 있다.

```c++
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
```

이 함수는 `count` bytes 만큼의 page-locked host memory를 할당하며, 할당된 메모리는 pinned memory와 동일하게 `cudaFreeHost`를 통해 해제할 수 있다. `flags` 파라미터를 인자로 받는데, 여기에 전달되는 플래그에 따라 할당된 메모리의 특별한 속성이 지정된다. 지정할 수 있는 플래그는 다음과 같다.

- `cudaHostAllocDefault` : 이 플래그를 지정하면 `cudaMallocHost`와 동일한 동작을 수행한다. 즉, pinned memory(without mapped)를 할당한다.
- `cudaHostAllocPortable` : 모든 CUDA context에서 사용할 수 있는 pinned memory를 할당한다.
- `cudaHostAllocWriteCombined` : write-combined memory를 할당한다고 한다. host 측에서 read 효율이 매우 낮기 때문에 host에서는 write만 수행하고 device에서는 read만 수행하는 경우에 사용하기 적합하다. 
- `cudaHostAllocMapped` : zero-copy memory를 할당한다 (pinned + mapped host memory).

> 각 플래그에 따라 할당되는 메모리 특징은 [Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)를 참조

지금 관심있는 플래그는 오직 `cudaHostAllocMapped`이다. 이 플래그를 사용하여 zero-copy memory를 할당한 뒤, 이 메모리를 device 측에서 사용하려면 이 pinned host memory에 매핑되는 device pointer를 알아내야 한다. 이때 사용하는 함수가 바로 `cudaHostGetDevicePointer` 이다.
```c++
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
```

이 함수는 device에서 mapped, pinned host memory에 참조할 수 있는 device pointer를 리턴한다. 만약 device가 mapped, pinned host memory를 지원하지 않는다면 이 함수는 에러를 리턴한다. `flags`에는 현재 0만 전달 가능하다.

Zero-copy를 사용해서 read/write를 빈번하게 수행하면 성능이 크게 저하된다. Mapped memory에 대한 모든 memory transaction은 PCIe Bus를 통과해야되기 때문에 global memory와 비교했을 때 상당한 latency가 추가된다. Global memory와 zero-copy memory 간의 성능 비교는 [vector_add_zerocopy.cu](/cuda/code/vector_add/vector_add_zerocopy.cu) 코드를 통해 확인할 수 있다. 아래 코드는 [vector_add_zerocopy.cu](/cuda/code/vector_add/vector_add_zerocopy.cu)의 일부분을 간단히 요약한 것이다.

```c++
...
/*************** Case 2: using zero-copy memory ***************/
// allocate zero-copy memory
unsigned int flags = cudaHostAllocMapped;
cudaHostAlloc(&h_a, bytes, flags);
cudaHostAlloc(&h_b, bytes, flags);

// initialize data at host side
initVector(h_a, num_elements);
initVector(h_b, num_elements);
(void*)memset(host_ref, 0, bytes);
(void*)memset(gpu_ref, 0, bytes);

// pass the pointer to device
cudaHostGetDevicePointer(&d_a, h_a, 0);
cudaHostGetDevicePointer(&d_b, h_b, 0);

// launch kernel with zero-copy memory
vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);

// copy kernel result back to host side
cudaMemcpy(gpu_ref, d_c, bytes, cudaMemcpyDeviceToHost);

cudaFreeHost(h_a);
cudaFreeHost(h_b);
...
```

host memory는 `cudaHostAlloc`을 통해서 할당하고(해제는 `cudaFreeHost`를 통해), 커널 함수 내에서 사용하기 위해 `cudaHostGetDevicePointer`를 통해서 device pointer를 알아내는 것 이외에는 global memory를 사용하는 것과 동일하다.

이 코드를 컴파일하고, 실행하면 아래와 같은 출력 결과를 얻을 수 있다. 아래 결과는 벡터 크기를 $2^{22}$ 로 지정헀을 때의 결과이다.
```
$ ./vector_add_zerocopy
> Vector size: 4194304 elements bytes:  16 MB
> vectorAddKernel(global memory)    Elapsed Time: 0.076679 ms
> vectorAddKernel(zero-copy memory) Elapsed Time: 1.303188 ms
Done
```

측정 결과, zero-copy memory를 사용했을 때의 속도가 훨씬 더 느리다는 것을 볼 수 있다.

하지만, 벡터의 크기를 $2^{10}$ 정도로 줄여서 테스트해보면 조금 다른 결과를 얻을 수 있다.
```
$ ./vector_add_zerocopy 10
> Vector size: 1024 elements bytes:   4 KB
> vectorAddKernel(global memory)    Elapsed Time: 0.003485 ms
> vectorAddKernel(zero-copy memory) Elapsed Time: 0.005637 ms
Done
```

벡터의 크기를 다르게 헀을 때 global memory와 zero-copy memory에서의 속도는 다음과 같이 측정된다.

|Size|Device Memory|Zero-copy Memory|
|--|--|--|
|1K|0.0037 ms|0.0053 ms|
|4K|0.0035 ms|0.0056 ms|
|16K|0.0035 ms|0.0069 ms|
|64K|0.0035 ms|0.012 ms|
|256K|0.004 ms|0.023 ms|
|1M|0.005 ms|0.096 ms|
|4M|0.022 ms|0.364 ms|
|16M|0.077 ms|1.303 ms|
|64M|0.293 ms|5.155 ms|

이 결과를 통해 host와 device 간 전달되는 데이터 크기가 작으면 zero-copy memory는 사용할만하다는 것을 알 수 있다. 특히, 명시적인 데이터 복사를 프로그래밍하지 않아도 되기 때문에 코드가 깔끔해지고 성능도 꽤 준수한 편이다. 하지만 크기가 큰 데이터의 경우 zero-copy memory를 사용하면 global memory와 비교했을 때 성능이 매우 안좋다는 것을 알 수 있다.

> Heterogeneous system architecture는 `Integrated`와 `Discrete`로 분류할 수 있다.
>
> Integrated 아키텍처에서 CPU와 GPU는 single die에 있으며 물리적으로 main memory를 공유한다. 이러한 아키텍처에서 zero-copy memory는 더 좋은 성능과 편의성을 얻을 수 있는데, PCIe bus를 통한 복사가 필요하지 않기 때문이다.
>
> 반면 discrete 아키텍처는 host와 device가 분리되어 있고, PCIe bus를 통해 연결되어 있다. 따라서, 특별한 경우에서만 zero-copy memory가 유용하다고 볼 수 있다.
>
> Mapped pinned memory는 host와 device 간에 공유되기 때문에 잠재적인 race condition이 발생할 수 있다. 따라서, 여러 스레드에서 동일한 메모리에 액세스하는 경우, 동기화가 필요하다.
>
> 또한, zero-copy memory는 high-latency를 가지므로 과도하게 사용하지 않도록 유의해야 한다.

<br>

# Unified Virtual Addressing (UVA)

Compute capability 2.0 이상의 device에서 **Unified Virtual Addressing (UVA)** 이라는 특별한 addressing mode를 지원한다 (64-bit process에서). UVA에서 host memory와 device memory는 하나의 virtual address space를 공유하며, 아래 그림과 같이 표현할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3SgDH%2FbtrZPT1x4V5%2FQGsK0vRyinhcDbPtEgqSz1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

UVA 이전에는 host memory를 참조하는 포인터와 device memory를 참조하는 포인터를 따로 관리했다. 하지만 UVA를 사용하면 포인터에 의해서 참조되는 메모리 공간은 어플리케이션 레벨에서 하나로 관리할 수 있다.

UVA에서 `cudaHostAlloc`를 통해 할당되는 pinned host memory는 동일한 host & device pointer를 갖게 된다. 즉, 이 API를 통해 할당받은 메모리의 주소를 커널 함수에 그대로 전달하여 device에서 사용할 수 있다는 것을 의미한다. [Zero-copy Memory](#zero-copy-memory)에서는 `cudaHostAlloc`으로 할당한 pinned host memory를 사용하려면 `cudaHostGetDevicePointer`를 사용하여 device pointer를 알아내야 했고, 이 포인터를 커널 함수에 전달했었다.

따라서, UVA를 사용하면 할당한 pinned host memory를 device에서 사용하기 위해 device pointer를 얻는 과정이 필요없다. 즉, 아래와 같이 그냥 `cudaHostAlloc`으로 할당하여 얻은 메모리 주소를 그대로 커널 함수에 전달하여 사용할 수 있다. [vector_add_zerocopy.cu](/cuda/code/vector_add/vector_add_zerocopy.cu) 코드에서 `cudaHostGetDevicePointer`를 사용하는 부분을 제거하고, 커널 함수에 `h_a`와 `h_b`를 그대로 전달해주기만 하면 된다.
```c++
...
// allocate zero-copy memory
unsigned int flags = cudaHostAllocMapped;
cudaHostAlloc(&h_a, bytes, flags);
cudaHostAlloc(&h_b, bytes, flags);

// initialize data at host side
initVector(h_a, num_elements);
initVector(h_b, num_elements);
(void*)memset(host_ref, 0, bytes);
(void*)memset(gpu_ref, 0, bytes);

// launch kernel with zero-copy memory
vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(h_a, h_b, d_c, num_elements);
...
```

똑같이 zero-copy memory를 사용하지만, UVA를 지원하면 코드가 더욱 깔끔해진다. 물론 성능은 device pointer를 사용하는 것과 동일하다.

<br>

# Unified Memory

**Unified Memory**는 CUDA 6.0에서 처음 도입되었으며, CUDA 프로그래밍 모델에서 메모리 관리를 더욱 간편하게 하기 위해서 도입되었다. Unified Memory는 managed memory pool을 생성하는데, 이 memory pool에서 할당된 메모리는 CPU와 GPU에서 동일한 메모리 주소(pointer)를 통해 액세스 가능하다. Unified Memory 내부에서는 시스템이 unified memory space에서의 host와 device 간 data migration을 자동으로 수행한다. 즉, `cudaMemcpy`를 통한 명시적인 데이터 전달이  Unified Memory에서는 숨겨져있다고 볼 수 있다.

> Host<->device 간 메모리 전달이 아예 없어지는 것이 아니다. 단순히 코드 상에서 눈에 보이지 않을 뿐이다 (명시적인 memory copy call이 필요없음).

간단히 정리하면 unified memory는 zero-copy memory에서 발생하는 성능의 저하 없이 `cudaMemcpy*()`를 통한 명시적인 데이터 이동을 제거한다. 물론, 그 이면에서 데이터 이동은 여전히 존재하기 때문에 프로그램의 실행 시간이 줄어들지는 않으며, 더 간단하고 유지 관리가 편한 코드를 작성할 수 있다는 이점이 있다.

> Unified Memory는 UVA(Unified Virtual Addressing)의 서포트에 의존하지만, 둘은 완전히 다른 기술이다. UVA는 시스템의 모든 프로세서에 single virtual memory address를 제공하지만, 하나의 물리적 공간에서 다른 물리적 공간으로 data migration을 수행하지는 않는다.

> Unified Memory는 zero-copy memory와 개념적으로 유사한 "single-pointer-to-data" 모델을 제공한다. 하지만 zero-copy memory는 host memory에 할당된다. 따라서, PCIe bus의 high latency로 인해 커널의 성능이 좋지 않다.

**Managed memory**는 기본 시스템에서 자동으로 관리되는 Unified Memory allocation을 지칭한다. 커널에서는 시스템에 의해서 관리되는 managed memory와 CUDA API를 통해 명시적으로 할당되고 전송되는 un-managed memory의 두 가지 타입의 메모리를 모두 사용할 수 있다. Device memory에 유효한 모든 CUDA operation은 managed memory에도 유효하다. 주요한 차이점은 managed memory는 host도 참조하고 액세스할 수 있다는 것이다.

Managed memory는 device global memory와 마찬가지로 정적 또는 동적으로 할당될 수 있다. 만약 변수를 선언할 때 `__managed__`를 추가해주면 managed variable을 선언할 수 있다. 이 변수는 file-scope와 global-scope 에서만 선언 가능하며 host 또는 device code에서 직접 액세스할 수 있다.
```c++
__device__ __managed__ int y;
```

동적으로 managed memory를 할당하려면 `cudaMallocManaged`를 사용하면 된다.
```c++
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int floags = cudaMemAttachGlobal);
```

이렇게 할당된 managed memory는 `cudaFree`를 통해 해제한다.

[vector_add_unified_memory.cu](/cuda/code/vector_add/vector_add_unified_memory.cu)에서 unified memory를 어떻게 사용하는지 살펴볼 수 있다. 아래 코드는 해당 코드의 일부분을 간략히 나타낸 것이다.

```c++
...
// allocate unified memory
float *um_a, *um_b, *um_c;
cudaMallocManaged(&um_a, bytes);
cudaMallocManaged(&um_b, bytes);
cudaMallocManaged(&um_c, bytes);

// initialize data at host side
initVector(um_a, num_elements);
initVector(um_b, num_elements);
(void*)memset(host_ref, 0, bytes);

// add vector at host side for result check
vectorAddOnHost(um_a, um_b, host_ref, num_elements);

// launch kernel with unified memory
vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(um_a, um_b, um_c, num_elements);

// check device results
checkResult(host_ref, um_c, num_elements);

// free memory
cudaFree(um_a);
cudaFree(um_b);
cudaFree(um_c);
...
```

코드를 살펴보면, `cudaMallocManaged`로 할당한 메모리의 주소는 host와 device 모두에서 동일한 포인터로 사용 가능하다. Host에서 사용되다가 device에서 사용되면, 시스템이 자동으로 data migration(HtoD)를 수행한다. 이를 통해서 코드를 더욱 간략하게 작성할 수 있으며 유지보수도 편하다.

[vector_add_unified_memory.cu](/cuda/code/vector_add/vector_add_unified_memory.cu)를 컴파일하고, `nsight system`으로 프로그램을 프로파일링해보면 아래와 같은 출력을 얻을 수 있다.
```
$ sudo nsys profile --stats=true ./vector_add_unified_memory
...
[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)              Operation            
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ---------------------------------
     36.0        2,869,598    689      4,164.9      1,951.0      1,566     58,560      8,434.4  [CUDA Unified Memory memcpy HtoD]
     32.6        2,597,997      2  1,298,998.5  1,298,998.5  1,293,991  1,304,006      7,081.7  [CUDA memcpy HtoD]               
     15.8        1,263,110      1  1,263,110.0  1,263,110.0  1,263,110  1,263,110          0.0  [CUDA memcpy DtoH]               
     15.6        1,239,892     96     12,915.5      2,751.5      1,023    104,001     21,746.2  [CUDA Unified Memory memcpy DtoH]
...
```

Unified memory에서 발생하는 암시적인 memcpy을 확인할 수 있다.

> [Matrix Addition with Unified Memory](/cuda/study/13_matrix_addition_with_unified_memory.md)에서 unified memory에 대해 조금 더 살펴볼 수 있다

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
- [NVIDIA CUDA Documentation: Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming)