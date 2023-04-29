# Table of Contents

- [Table of Contents](#table-of-contents)
- [Stream Synchronization Behavior](#stream-synchronization-behavior)
  - [Default Stream](#default-stream)
  - [Legacy Default Stream](#legacy-default-stream)
  - [Per-Thread Default Stream](#per-thread-default-stream)
  - [For Multi-Threads](#for-multi-threads)
- [References](#references)

<br>

# Stream Synchronization Behavior

## Default Stream

`cudaStream_t`로 0이 전달되거나 암시적으로 스트림에서 동작하는 API에서 사용되는 `default stream`은 `legacy` 또는 `per-thread` synchronization behavior을 갖도록 구성할 수 있다.

`nvcc` 옵션인 `--default-stream`을 사용하여 컴파일 단위별로 동작을 제어할 수 있다. 또는, CUDA 헤더를 include 하기 전에 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 매크로를 정의하여 per-thread 동작을 활성화할 수 있다. 어떤 방법이든 `CUDA_API_THREAD_DEFAULT_STREAM` 매크로가 컴파일 단위에서 정의된다.

## Legacy Default Stream

Lecagy default stream은 아래에서 설명하는 non-blocking streams를 제외하고 동일한 `CUcontext`의 다른 모든 스트림과 동기화되는 implicit stream이다 (Runtime API만 사용하는 어플리케이션에는 device 당 하나의 컨텍스트만 존재한다). 일반적으로 우리가 알고 있는 default stream과 동일하다.

예를 들어, 아래 코드를 컴파일하고 `nsight system`으로 어떤 시퀀스로 실행되는지 살펴보자.
```c++
__global__
void kernel(float* x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
 
    for (int i = tid; i < n; i += stride) {
        x[i] = sqrt(pow(3.14159, i));
    }
}
 
int main()
{
    const int N = 1 << 22;
 
    float* data1, *data2, *data3;
    cudaMalloc(&data1, sizeof(float) * N);
    cudaMalloc(&data2, sizeof(float) * N);
    cudaMalloc(&data3, sizeof(float) * N);
 
    cudaStream_t s;
    cudaStreamCreate(&s);
 
    kernel<<<1, 64, 0, s>>>(data1, N);
    kernel<<<1, 64>>>(data2, N);
    kernel<<<1, 64, 0, s>>>(data3, N);
 
    cudaDeviceSynchronize();
 
    cudaFree(data1);
    cudaFree(data2);
    cudaFree(data3);
    cudaDeviceReset();
}
```
코드 중간에 커널을 호출하는 부분을 살펴보면, 첫 번째와 세 번째 커널은 명시적으로 default가 아닌 스트림이 지정되었고 두 번째 커널은 스트림이 지정되지 않았기 때문에 default stream에서 실행된다. Default stream은 다른 모든 스트림을 블로킹(blocking)하기 때문에 첫 번째 커널 실행이 끝나기 전에 두 번째 커널이 실행될 수 없고, 두 번째 커널은 세 번째 커널 실행을 블로킹한다.

위 코드를 `nsight system`으로 프로파일링한 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbXWyJe%2FbtsddtCnQpG%2FyWv31QtT0itdE83LkwwQGk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

예상한 것과 같이 default 스트림의 블로킹 동작으로 인해 각 커널이 그냥 순차적으로 실행된 것을 볼 수 있다. Default stream은 `nsight system` 내에서 `Default stream 7`로 표시되고 있다.

반면, non-blocking streams는 legacy stream과 동기화하기 않는다. Non-blocking stream은 스트림 생성 API에 `cudaStreamNonBlocking` 플래그를 사용하여 생성할 수 있다. 다음 코드를 통해 커널의 동작을 살펴보자.
```c++
__global__
void kernel(float* x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
 
    for (int i = tid; i < n; i += stride) {
        x[i] = sqrt(pow(3.14159, i));
    }
}

int main()
{
    const int N = 1 << 22;
 
    float* data1, *data2, *data3;
    cudaMalloc(&data1, sizeof(float) * N);
    cudaMalloc(&data2, sizeof(float) * N);
    cudaMalloc(&data3, sizeof(float) * N);
 
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
 
    kernel<<<1, 64, 0, s>>>(data1, N);
    kernel<<<1, 64>>>(data2, N);
    kernel<<<1, 64, 0, s>>>(data3, N);
 
    cudaDeviceSynchronize();
 
    cudaFree(data1);
    cudaFree(data2);
    cudaFree(data3);
    cudaDeviceReset();
}
```

위 코드를 컴파일한 뒤, `nsight system`으로 프로파일링한 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FS7ToT%2Fbtsdc0tUcH7%2FrIc4n4bXqKn0qQLT0IOKtk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

명시적으로 생성된 스트림이 non-blocking으로 생성되었기 때문에 해당 스트림들은 default stream에 non-blocking 동작을 갖는다. 따라서, 해당 스트림 사이에 default stream으로 실행되는 작업이 있더라도 블로킹되지 않고 동시에 실행된 것을 확인할 수 있다.

## Per-Thread Default Stream

Per-thread default stream은 스레드와 `CUcontext` 모두에게 local인 implicit stream이며, 이들은 어떤 스트림과도 동기화하지 않는다 (즉, default stream이 명시적으로 생성된 스트림처럼 동작한다). Per-thread default stream는 non-blocking stream이 아니며, 만약 프로그램 내에서 legacy default stream과 같이 사용되면 per-thread default stream은 legacy stream에 동기화한다.

예제 코드를 통해서 조금 더 자세히 살펴보자. 아래 코드는 8개의 스트림을 생성하고, for 루프를 통해 각 스트림에서 커널을 실행하는데 각 스트림에서 실행되는 커널 사이에 default stream에서 실행되는 커널을 하나씩 추가한다.

```c++
__global__
void kernel(float* x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
 
    for (int i = tid; i < n; i += stride) {
        x[i] = sqrt(pow(3.14159, i));
    }
}
 
int main()
{
    const int N = 1 << 22;
    const int num_streams = 8;
 
    cudaStream_t streams[num_streams];
    float* data[num_streams];
 
    for (int stream = 0; stream < num_streams; stream++) {
        cudaStreamCreate(&streams[stream]);
        cudaMalloc(&data[stream], sizeof(float) * N);
 
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[stream]>>>(data[stream], N);
 
        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }
 
    for (int stream = 0; stream < num_streams; stream++) {
        cudaFree(data[stream]);
    }
 
    cudaDeviceReset();
}
```

먼저 이 코드를 legacy default stream 동작으로 컴파일하여 `nsight system`으로 프로파일링하면 아래의 결과를 얻을 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHrQ7n%2FbtsdeZ1AnLD%2FzD7ot4B5CWOb6V9E6Oso5K%2Fimg.png" width=800px style="display: block; margin: 0 auto"/>

각 스트림에서의 커널 실행 사이마다 default stream에서의 커널 실행이 포함되어 있기 때문에, 위와 같이 모든 커널들이 순차적으로 실행되는 것을 확인할 수 있다. 이와 같은 동작이 우리가 익숙한 default stream의 동작이다 (legacy default stream은 다른 모든 스트림에 동기화된다).

이번에는 동일한 코드를 `nvcc`에 `--default-stream per-thread` 옵션을 추가하여 컴파일한 뒤, `nsight systeam`으로 프로파일링한 결과를 살펴보자.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbM0BUN%2FbtsdhAtIw7t%2FLhjvkkry2Ng6E3VeGSMDu0%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

8개의 명시적 스트림과 default 스트림에서의 커널 실행이 모두 동시에 수행되는 것을 확인할 수 있다. 위의 결과와 비교했을 때, 동일한 코드지만 스트림의 동작을 per-thread로 설정하면 default stream의 동작이 달라진다는 것을 확인할 수 있다. 즉, per-thread default stream으로 설정하면 default stream이 명시적으로 생성되는 스트림과 동일한 동작을 갖게 된다. Legacy default stream인 경우에는 default stream이 `Default stream 7`로 표시되었지만, per-thread default stream인 경우에는 `Stream 14`로 표시되고 있다는 것을 눈여겨 볼 필요가 있다.

## For Multi-Threads

이번에는 multi-thread 환경에서의 각 스트림 동작을 아래 코드를 통해서 살펴보자. 아래 코드는 8개의 host 스레드를 생성하고 각 스레드에서 implicit stream으로 커널을 실행한다.
```c++
#include <thread>
 
__global__
void kernel(float* x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
 
    for (int i = tid; i < n; i += stride) {
        x[i] = sqrt(pow(3.14159, i));
    }
}
 
void launch_kernel(float* data, int n)
{
    cudaMalloc(&data, sizeof(float) * n);
 
    kernel<<<1, 64>>>(data, n);
 
    cudaStreamSynchronize(0);
}
 
int main()
{
    const int N = 1 << 22;
    const int num_threads = 8;
 
    float* data[num_threads];
    std::thread threads[num_threads];
 
    for (int i = 0; i < num_threads; i++) {
        threads[i] = std::thread(launch_kernel, data[i], N);
    }
 
    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
        cudaFree(data[i]);
    }
 
    cudaDeviceReset();
}
```

먼저 위 코드를 legacy default stream으로 동작하도록 컴파일하여 프로파일링한 결과를 살펴보자.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbnnm67%2FbtsddZA0Dmd%2FGpKRxtFMk8c3Z2YPQ9yPo1%2Fimg.png" width=800px style="display: block; margin: 0 auto"/>

각 스레드에서 모든 커널이 동일한 default stream에서 동작하기 때문에, 모든 커널이 순차적으로 수행된다는 것을 확인할 수 있다. 명시적인 스트림이 없기 때문에 `nsight system`에서 스트림에 대한 정보는 따로 표시되지 않는다는 것을 알 수 있다.

반면, per-thread로 동작하도록 컴파일한 뒤 프로파일링한 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcKp6Iy%2Fbtsdc01KOuv%2FmQXGEQP5RMFHfF0phZkWa0%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

각 스레드에서 실행되는 커널들은 자신만의 default stream에서 수행되기 때문에 모든 커널들이 동시에 수행된다는 것을 확인할 수 있다. 이전 결과와는 다르게 스트림 정보도 나타나는 것을 볼 수 있다.


<br>

# References

- [NVIDIA CUDA Documentation: Stream Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior)