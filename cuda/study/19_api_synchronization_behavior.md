# Table of Contents

- [Table of Contents](#table-of-contents)
- [API Synchronization Behavior](#api-synchronization-behavior)
- [Memcpy](#memcpy)
  - [Synchronous](#synchronous)
  - [Asynchronous](#asynchronous)
- [Memset](#memset)
- [Kernel Launches](#kernel-launches)
- [References](#references)

<br>

# API Synchronization Behavior

이번 포스팅에서는 Runtime API의 동기화 동작에 대해서 자세히 살펴본다.

Runtime API는 동기 또는 비동기 형태의 memcpy / memset 함수를 제공하며, 비동기로 동작하는 함수의 경우에는 `async`라는 접미사가 붙는다. 사실, `async`라는 이름이 붙었다고 무조건 비동기로 동작하지 않으며, **함수에 전달된 인자에 따라서 각 함수가 동작하는 방식이 다르므로 주의**해야 한다.

<br>

# Memcpy

각 memcpy 함수는 아래 정의에 따라 동기(synchronous) 또는 비동기(asynchronous)로 카테고리화된다.

<br>

## Synchronous

**(synchronous 버전 API의 경우)**

1. `Pageable host memory`로부터 `device memory`로의 복사의 경우, 복사가 시작되기 전에 스트림 동기화(stream sync)가 수행된다. Device memory로의 DMA transfer를 위해 pageable buffer가 staging memory로 복사되면 함수는 즉시 반환하지만, 최종 목적지(destination)으로의 DMA가 완료되지 않았을 수도 있다.
2. `Pinned host memory`로부터 `device memory`로의 복사의 경우, 함수는 host에 대해 동기화된다.
3. `Device memory`로부터 `pageable` 또는 `pinned host memory`로의 복사의 경우, 함수는 복사가 완료된 후에만 제어권을 반환한다.
4. `Device memory`로부터 `device memory`로의 복사의 경우, host측에 대한 어떠한 동기화도 수행되지 않는다 (비동기로 동작).
5. `Any host memory`로부터 `any host memory`로의 복사의 경우, 함수는 host에 대해 완전히 동기화된다.

각 정의에 따라 정확히 어떻게 동작하는지 `nsight system`을 통해 자세히 살펴보자. 각 복사에서 제어권이 host로 반환되자마자 `nvtx`를 통해 마킹하여 반환 시점을 확인했다.

### Transfer from pageable host memory to device memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    a = (double*)malloc(sizeof(double) * n);
    cudaMalloc(&b, sizeof(double) * n);
 
    cudaMemcpy(b, a, sizeof(double)*n, cudaMemcpyHostToDevice);
    nvtxMarkA("marking");
    
    free(a);
    cudaFree(b);
}
```

위 코드의 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FboW4PT%2Fbtsdc8ry7Zv%2F3TTFHKkooYfj6f0kQPwHLK%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

일반적인 `cudaMemcpy`의 동작을 생각한다면 무조건 host에 동기화되므로, 복사가 완전히 끝난고 반환될 것이라고 예상된다. 하지만, `marking`이 찍힌 위치를 보면 완전히 복사가 끝나기도 전에 host로 리턴되었다는 것을 알 수 있다. 즉, pageable buffer가 staging memory로 복사된 이후에 함수가 바로 리턴되었다는 것을 보여준다.

### Transfer from pinned host memory to device memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaMallocHost(&a, sizeof(double) * n);
    cudaMalloc(&b, sizeof(double) * n);

    cudaMemcpy(b, a, sizeof(double)*n, cudaMemcpyHostToDevice);
    nvtxMarkA("marking");
    
    cudaFreeHost(a);
    cudaFree(b);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fclz9Q4%2FbtsdbPTCtJU%2FADAvVvzrmlN0kjI4v8SBCK%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

이번에는 `marking`이 GPU에서 복사가 완료된 이후에 찍힌 것을 확인할 수 있다. 즉, host에 대해서 동기화된다는 것을 보여준다.

### Transfer from device memory to pageable or pinned host memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaMalloc(&a, sizeof(double) * n);
    b = (double*)malloc(sizeof(double) * n);
 
    cudaMemcpy(b, a, sizeof(double)*n, cudaMemcpyDeviceToHost);
    nvtxMarkA("marking");
    
    cudaFree(a);
    free(b);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fblclte%2FbtsdeHzKzja%2FvVlKk8KFHo8FolDDZ8Ae81%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

`Device memory`로부터 `pageable memory`로의 전송하는 경우에는 복사가 완료된 이후에만 반환된다는 것을 보여준다.

### Transfer from device memory to device memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaMalloc(&a, sizeof(double) * n);
    cudaMalloc(&b, sizeof(double) * n);
 
    cudaMemcpy(b, a, sizeof(double)*n, cudaMemcpyDeviceToDevice);
    nvtxMarkA("marking");
    
    cudaFree(a);
    cudaFree(b);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8tZlk%2FbtsdcJr311P%2FKoPRK84mBqu4tg9mHBuTv1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

위 결과는 `cudaMemcpy()`가 무조건 동기로 동작하지 않는다는 것을 명확하게 보여준다. 즉, `cudaMemcpy()`를 통한 DtoD 복사는 host에 대한 동기화가 전혀 수행되지 않는다.

### Transfer from any host memory to any host memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaMallocHost(&a, sizeof(double) * n);
    cudaMallocHost(&b, sizeof(double) * n);
 
    cudaMemcpy(b, a, sizeof(double)*n, cudaMemcpyHostToHost);
    nvtxMarkA("marking");
    
    cudaFreeHost(a);
    cudaFreeHost(b);
}
```

`a`와 `b`를 모두 `pinned host memory`로 할당하여 테스트했다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcbSfp0%2FbtsdcryszTG%2FVXJTo0YkZ8bkctcQ7rUFo1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

HtoH 복사는 GPU가 관여하지 않으므로 CUDA Trace 결과 자체가 없다. 따라서, host에 완전히 동기화되어 동작한다는 것을 짐작할 수 있다.

<br>

## Asynchronous

**(asynchronous 버전 API의 경우)**

1. `Device memory`로부터 `pageable host memory`로의 복사의 경우, 함수는 복사가 완료된 후에만 제어권을 반환한다.
2. `Any host memory`로부터 `any host memory`로의 복사의 경우, 함수는 host에 대해 완전히 동기화된다.
3. 만약 `pageable memory`가 먼저 `pinned memory`로 스테이징해야 하는 경우, driver는 스트림과 동기화하고 `pinned memory`로의 복사를 수행할 수 있다.
   <br>(pageable memory가 pinned memory로 스테이징된 이후에는 host에 비동기로 동작하는 것으로 추측됨. [Transfer from device memory to pageable host memory](#transfer-from-device-memory-to-pageable-host-memory)에서 확인).
4. 1~3번의 경우를 제외한 모든 복사의 경우, 함수는 완전히 비동기로 동작한다.

마찬가지로 비동기 동작에 대해서도 자세히 살펴보자.

### Transfer from device memory to pageable host memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    cudaMalloc(&a, sizeof(double) * n);
    b = (double*)malloc(sizeof(double) * n);
 
    cudaMemcpyAsync(b, a, sizeof(double)*n, cudaMemcpyDeviceToHost, stream);
    nvtxMarkA("marking");
    
    cudaFree(a);
    free(b);
    cudaStreamDestroy(stream);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdBvJX%2FbtsdaZhTmVD%2F9xK00yYsALwcw4uno3VhfK%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

`cudaMemcpyAsync()`가 무조건 비동기로 동작하는 것이 아니라는 것을 보여주는 전형적인 예시이다. API 문서나 CUDA 문서에서도 나와있지만, HtoD 또는 DtoH에 대한 `cudaMemcpyAsync()`가 비동기로 동작하려면 반드시 `pinned host memory`를 사용해야 한다. 여기서는 `pageable host memory`를 사용했기 때문에 비동기가 아닌 동기로 동작하며, 따라서 복사가 완료된 이후에 host로 반환된다.

### Transfer from any host memory to any host memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    cudaMallocHost(&a, sizeof(double) * n);
    cudaMallocHost(&b, sizeof(double) * n);
 
    cudaMemcpyAsync(b, a, sizeof(double)*n, cudaMemcpyHostToHost, stream);
    nvtxMarkA("marking");
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaStreamDestroy(stream);
}
```

모두 `pinned host memory`로 할당하여 복사 테스트를 수행했다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbNNZDh%2Fbtsdc7MY2b7%2FIMzDhMMmUebOkjGINy5H9K%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

마찬가지로 HtoH 메모리 복사이므로 GPU가 관여하지 않는다. 따라서, host에 동기화될 것이라고 추측할 수 있다.

### Transfer from pinned host memory to device memory

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    cudaMallocHost(&a, sizeof(double) * n);
    cudaMalloc(&b, sizeof(double) * n);
 
    cudaMemcpyAsync(b, a, sizeof(double)*n, cudaMemcpyHostToDevice, stream);
    nvtxMarkA("marking");
    
    cudaFreeHost(a);
    cudaFree(b);
    cudaStreamDestroy(stream);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcCfP6V%2FbtsdcKLiu5J%2FBzWKtlcnKGPJqVtnc3yYC0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

전형적인 비동기 메모리 복사 예시를 보여준다. `cudaMemcpyAsync()`는 거의 호출되자마자 제어권을 다시 host로 반환하며, 비동기로 메모리 복사가 수행된다 (복사가 끝나기 전에 `marking`이 찍혀있다).

### Transfer from device memory to pinned host memor

바로 위의 경우와 반대 상황이다.

```c++
#include <nvtx3/nvToolsExt.h>
 
int main()
{
    int n = 1 << 22;
    double *a, *b;
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    cudaMalloc(&a, sizeof(double) * n);
    cudaMallocHost(&b, sizeof(double) * n);
 
    cudaMemcpyAsync(b, a, sizeof(double)*n, cudaMemcpyDeviceToHost, stream);
    nvtxMarkA("marking");
    
    cudaFree(a);
    cudaFreeHost(b);
    cudaStreamDestroy(stream);
}
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMhwQn%2Fbtsdazqfxl6%2FWbpkjoKodh7VNXu4GgGiQk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

마찬가지로 복사가 완료되기 전에 host측으로 제어권이 반환되었으며, 따라서, `marking`이 복사가 완료되기 전에 찍혀있는 것을 확인할 수 있다.

<br>

# Memset

`cudaMemset` 함수는 target memory가 `pinned host memory`인 경우를 제외하고, host에 대해 비동기로 동작한다. `async` 버전은 항상 host와 비동기로 동작한다.

> `cudaMemset()`을 간단히 테스트했을 때, 모든 host memory에 대해서 동기로 동작하는 것으로 관찰되었다... Host 메모리를 굳이 CUDA API로 다룰 필요는 없어서 더 자세히 테스트해보지는 않았는데, 문서가 잘못된 것인지 테스트가 잘못된 것인지 잘 모르겠다.

<br>

# Kernel Launches

Kernel lauches는 항상 host에 대해 비동기로 동작한다.

<br>

# References

- [NVIDIA CUDA Documentation: API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior)