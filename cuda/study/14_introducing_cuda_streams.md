# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introducing Streams](#introducing-streams)
- [CUDA Streams](#cuda-streams)
  - [Stream Example: Vector Addition](#stream-example-vector-addition)
- [Stream Scheduling](#stream-scheduling)
  - [False Dependencies](#false-dependencies)
  - [Hyper-Q](#hyper-q)
- [Stream Priorities](#stream-priorities)
- [Stream Synchronization](#stream-synchronization)
  - [Blocking and Non-Blocking Streams](#blocking-and-non-blocking-streams)
  - [Implicit Syncrhonization](#implicit-syncrhonization)
  - [Explicit Synchronization](#explicit-synchronization)
- [References](#references)

<br>

# Introducing Streams

일반적으로 CUDA 프로그래밍에서는 2가지 레벨의 동시성(concurrency)가 있다.

- Kernel level concurrency
- Grid level concurrency

**CUDA 스트림(stream)** 대해서 알지 못한다면, 지금까지 알고 있는 모든 것들은 GPU에서 수 많은 스레드들을 통해 병렬로 수행하는 하나의 task(or kernel)를 의미하는 **kernel level concurrency**에 해당된다.

이번 포스팅에서는 CUDA 스트림과 **grid level concurrency**에 대해서 알아본다. Grid level concurrency에서는 여러 커널들이 하나의 GPU 장치에서 동시에 수행되며, 이를 통해 더 좋은 device utilization을 얻을 수 있다.

**CUDA 스트림(stream)** 은 host code에서 실행한 순서대로 GPU device에서 실행되는 일련의 비동기 CUDA operations을 지칭한다. 조금 더 간단히 설명하면, 동시에 수행하는 커널들을 스트림을 통해 관리한다. 여러 커널들이 동시에 수행될 수 있으므로, 동시에 수행되는 커널들을 관리하기 위해 각각 스트림들이 별도로 존재할 수 있다. 스트림이 없을 때에는 단 하나의 work queue를 통해 host에서 실행된 순서대로 연산을 수행하지만, 스트림이 있을 때는 스트림 갯수만큼 work queue가 존재한다고 생각하면 쉽게 이해가 가능하다.

스트림에서 수행되는 연산들은 항상 host에 대해 비동기이다. CUDA 런타임은 스트림에서 실행되는 작업이 GPU 장치에서 실행되는 시기를 결정한다. Host에 대해 비동기이므로, 작업의 결과를 사용하기 전에 비동기 작업이 완료되었는지 CUDA API를 통해 확인해야 하며, 이는 프로그래머의 책임 영역이다. 스트림이 각각의 work queue라고 생각하면 된다고 했는데, 각 스트림 내에서 실행되는 작업들은 host에서 실행된 순서대로 수행된다. 하지만, 서로 다른 스트림의 작업들의 실행 순서에는 어떠한 제약도 없다는 점에 유의해야 한다.

특히, CUDA 스트림에서 대기 중(queued)인 모든 작업들은 비동기이므로 host system에서의 다른 작업과 오버랩될 수 있다. 따라서, GPU의 작업과 CPU에서는 작업을 서로 동시에 수행할 수 있어서 이러한 작업들의 비용을 숨길 수 있다.

CUDA 프로그래밍에서 GPU 연산의 전형적인 패턴은 다음과 같다.

1. Move input data from the host to the device
2. Execute a kernel on the device
3. Move the result from the device back to the host

많은 경우에서 커널을 수행하는 시간보다 데이터를 전송하는 시간이 더 크다. 스트림을 사용하면, 이러한 상황에서 CPU-GPU communication latency를 상당히 감소시킬 수 있다. Kernel execution과 data transfer를 여러 스트림으로 분리하여 실행시켜, kernel execution과 data transfer를 서로 오버랩하면 프로그램의 실행 시간은 훨씬 더 줄어든다.

CUDA API 함수는 일반적으로 동기(synchronous) 및 비동기(asynchronous) 함수로 분류할 수 있다. 동기 함수는 호출되면 그 작업을 완료할 때까지 host 스레드를 블로킹한다. 반면 비동기 함수는 호출된 직후 host 스레드에게 제어권을 곧바로 반환한다.

소트프웨어 관점에서 서로 다른 스트림의 CUDA 연산들은 동시에 실행된다. 하지만 물리적인 관점에서는 항상 동시에 실행되는 것은 아니다. PCIe bus 또는 각 SM에서의 리소스 가용성에 따라 서로 다른 CUDA 스트림이 완료될 때까지 대기할 수 있다.

<br>

# CUDA Streams

모든 CUDA 연산(kernel execution and data transfer)은 명시적이든 아니든 스트림에서 실행된다. 즉, 명시적으로 스트림을 사용하지 않더라도 우리는 사실 스트림을 사용하고 있던 것이며, 이 스트림을 default stream이라고 부른다. 따라서, CUDA 스트림에는 두 가지 타입의 스트림이 존재한다.

- Implicitly declared stream (NULL stream = default stream)
- Explicitly declared stream (non-NULL stream)

`NULL stream` 은 명시적으로 스트림을 지정하지 않은 kernel launch나 data transfer가 사용하는 `default stream`이다. 스트림을 처음 접한다면, 지금까지 사용했던 모든 커널 실행이나 데이터 전달에서는 `NULL stream` 또는 `default stream`을 사용했던 것이다.

반면, `non-null stream`은 명시적으로 생성되고 관리된다. 만약 CUDA 연산들을 서로 오버랩하고 싶다면, 반드시 non-null stream을 사용해야 한다. Non-null stream들을 사용하면, 아래의 작업들을 서로 동시에 수행하도록 할 수 있다.

- Computation on the host
- Computation on the device
- Memory transfers from the host to the device
- Memory transfers from the device to the host
- Memory transfers within the memory of a given device
- Memory transfers among devices

> 위와 같은 연산들 간 달성할 수 있는 동시성의 수준은 GPU device의 feature set 및 compute capability에 따라 다르다.

아래 코드는 default stream을 사용하는 코드의 일부이다.
```c++
cudaMemcpy(..., cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(...);
cudaMemcpy(..., cudaMemcpyDeviceToHost);
```
이러한 CUDA 프로그램의 동작을 이해하려면 항상 host와 device 관점을 모두 고려해야 한다. Device 관점에서 위의 3가지 작업(`cudaMemcpy`, `kernel`, `cudaMemcpy`)는 모두 host에 실행된 순서대로 default stream에서 수행된다. Host 관점에서 보면, 각 `cudaMemcpy`는 동기식이며 `cudaMemcpy`가 완료될 때까지 host는 블로킹된다. 반면, `kernel` 실행은 비동기이므로 `kernel`의 완료와 상관없이 바로 host 측으로 제어권이 반환된다. 

그러나, data transfer 또한 비동기로 실행될 수 있는데, 비동기로 실행하려면 반드시 CUDA stream을 사용해야 한다. CUDA 런타임 API는 아래의 비동기 버전의 `cudaMemcpy`를 제공한다.
```c++
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
```
위 함수를 호출할 때, 다섯 번째 파라미터로 명시적인 스트림을 전달하며, 기본값은 default stream으로 설정된다. 위 함수는 host에 대해 비동기로 동작하며, 즉, `cudaMemcpyAsync`는 호출되자마자 host thread로 제어권을 즉시 반환한다.

스트림을 사용하려면, 당연히 먼저 `non-null stream`을 아래 런타임 API를 통해 생성해주어야 한다.
```c++
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```

Non-default stream은 아래와 같이 생성할 수 있다.
```c++
cudaStream_t stream;
cudaStreamCreate(&stream);
```

`cudaStreamCreate`는 명시적으로 관리할 `non-null stream`을 생성하며, 생성된 스트림은 `pStream`으로 리턴된다. 이렇게 생성된 스트림은 `cudaMemcpyAsync`의 다섯 번째 stream 파라미터에 인자로 전달되거나 다른 비동기 CUDA API 함수에도 동일하게 전달될 수 있다. 비동기 CUDA API를 사용할 때 혼동될 수 있는 한 가지 문제는 이전에 실행된 비동기 작업에서 발생한 에러로 인해 에러 코드가 반환될 수 있다는 것이다. 즉, 에러 코드를 반환하는 API가 반드시 에러를 일으켰다고 볼 수 없다.

비동기 data transfer를 수행할 때는 반드시 pinned (page-locked=non-pagable) host memory를 사용해야 한다. Pinned memory는 `cudaMallocHost`(또는 `cudaHostAlloc`)을 통해 할당할 수 있다.

> [CUDA 문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlap-of-data-transfer-and-kernel-execution)에 의하면 host memory가 복사 연산에 관련된 경우, host memory는 반드시 page-locked 상태이어야 한다고 한다. 따라서, `cudaMemcpyAsync`를 통한 비동기 메모리 복사라도, non paged-locked host memory라면 동기식으로 동작할 수 있다.

> Pinned host memory로 할당하지 않으면, OS는 언제든지 이 메모리의 물리적인 위치를 이동시킬 수 있게 된다. 만약 pinned host memory가 아닌 host memory에 비동기 CUDA transfer가 수행되면, CUDA 런타임이 해당 메모리의 데이터를 전달하는 동안 OS가 host memory를 물리적으로 이동시킬 수 있다 (undefined behavior을 일으킨다).

명시적인 스트림에서 커널을 실행시키려면, execution configuration의 네 번째 파라미터에 명시적인 스트림을 제공하면 된다.
```c++
kernel<<<grid, block, sharedMemSize, stream>>>(...);
```

사용한 뒤에는 아래 API를 통해 stream 리소스를 해제한다.
```c++
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```

만약 어떤 스트림에 대해 `cudaStreamDestroy`가 호출되었지만 스트림에 대기 중인 작업이 있다면, `cudaStreamDestroy`는 호출된 즉시 반환되고 스트림의 모든 작업들이 완료된 후 스트림과 연관된 리소스들이 자동으로 해제된다.

모든 CUDA 스트림 연산들은 비동기이므로, CUDA API에서는 스트림의 모든 작업들이 완료되었는지 확인하기 위해 아래의 두 가지 API를 제공한다.
```c++
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
```

`cudaStreamSynchronize`는 전달된 스트림의 모든 연산들이 완료될 때까지 host를 블로킹한다.

`cudaStreamQuery`는 만약 스트림의 모든 작업이 완료되었는지 체크하며, 만약 완료되지 않았다면 host 스레드를 블로킹한다. 만약 모든 작업이 완료되었다면 `cudaSuccess`를 리턴하고, 아직 진행 중이거나 대기 중인 작업이 있다면 `cudaErrorNotReady`를 리턴한다.

## Stream Example: Vector Addition

> 전체 코드는 [vector_add_with_streams.cu](/cuda/code/streams/vector_add_with_streams.cu)을 참조

간단한 벡터 덧셈 예제를 통해 CUDA 스트림을 어떻게 사용하는지 살펴보고, default 스트림을 사용할 때와 비교해서 성능이 어떻게 달라지는지 살펴보자. [vector_add_with_streams.cu](/cuda/code/streams/vector_add_with_streams.cu)에서는 data transfer(HtoD, DtoH)와 kernel execution(vector addition)을 모두 측정하여 성능을 비교하였다.

먼저 default 스트림을 사용하는 버전의 코드는 아래와 같다.
```c++
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
vectorAddOnGPU<<<grid, block>>>(d_a, d_b, d_c, num_elements);
cudaMemcpy(gpu_ref, d_c, bytes, cudaMemcpyDeviceToHost);
```

입력으로 사용되는 데이터는 host에 위치하므로, `h_a`와 `h_b`의 데이터를 device인 `d_a`와 `d_b`로 먼저 복사해준다. 그리고 덧셈을 수행하는 커널을 실행한다. 그런 다음 device에서 계산된 결과를 다시 host 측으로 복사해주는 연산을 수행한다. `cudaMemcpy`는 host에 대해 동기식으로 동작하므로 복사가 완료될 때까지 host 스레드를 블로킹한다.

아래 코드는 non-default 스트림을 사용하는 버전의 코드이다. 예제 코드에서는 `NUM_STREMAS`의 값은 4로 지정되어 있으며, 따라서, 4개의 스트림을 사용하여 커널을 각각의 스트림에서 실행시킨다.
```c++
// vectorAddOnGPU kernel launch with streams
for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * num_elements_per_stream;
    cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
    vectorAddOnGPU<<<grid, block, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, num_elements_per_stream);
    cudaMemcpyAsync(gpu_ref + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);
}
// sync with streams
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

여기서 주의할 점은 4개의 스트림이 각각 처리할 데이터는 전체 데이터의 1/4이다. 따라서, 각 스트림에서 처리되는 데이터를 알맞게 분배하기 위해서 `offset`과 `bytes_per_stream`을 계산하여 사용한다. 4등분된 데이터는 각각의 스트림에서 동시에 수행된다.

Default 스트림과 non-default 스트림을 사용할 때의 차이점은 아래 그림에서 잘 보여주고 있다. 아래 그림의 경우에는 3개의 스트림을 사용하여 데이터를 3등분하여 동시에 수행한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqPREZ%2Fbtr1UT6M76q%2F3y6VknknyfJepaoNVhGZbk%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림을 보면 각 스트림의 data transfer 연산이 동시에 수행되지 않는다는 것을 확인할 수 있다. 정확히 말하자면, 각 스트림의 HtoD data transfer 연산과 DtoH data transfer 연산이 각각 동시에 수행되지 않는다. 이는 GPU가 PCIe bus를 공유하기 때문이다. PCIe bus는 HtoD, DtoH 방향으로 하나씩 존재하는데, 이 때문에 HtoD transfer와 DtoH transfer는 서로 동시에 수행될 수 있다.

전체 구현은 [vector_add_with_streams.cu](/cuda/code/streams/vector_add_with_streams.cu)를 참조하고, 이 코드를 컴파일하고 실행하면 다음과 같은 출력 결과를 얻을 수 있다.
```
$ ./vector_add_streams 
> Vector Addition(Manual) at device 0: NVIDIA GeForce RTX 3080
> with 16777216 elements
vectorAddOnGPU with default stream         : elapsed time 28.451649 ms
vectorAddOnGPU with non-default streams(4) : elapsed time 7.448704 ms
```

Default 스트림을 사용헀을 때는 약 28 ms가 걸리지만, 4개의 non-default 스트림을 사용하여 데이터를 분산시켜 여러 개의 스트림에서 동시에 수행했을 때는 약 7 ms가 걸린다. 대략 4배 정도의 속도 향상을 보여준다.

스트림을 사용하지 않는 것과 사용한 것의 차이는 `nsight system`을 통해 시각적으로 확인할 수 있다. 아래 그림은 각각 default 스트림과 non-default streams에서의 연산을 시각화해서 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdpAAJ0%2Fbtr1UtUWZJi%2FaFGuAE6Nqp2Wce1CvHi990%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

<br>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgQiUy%2Fbtr1TS1JPJq%2F7onNkMuasIZYxALX9KbNSk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

> `vectorAddOnGPU` 커널이 단순하기 때문에 데이터 복사 시간에 비해 상당히 짧은 실행 시간을 보여준다. 이처럼 GPU의 대부분 연산은 실제 연산보다 host와 device 간 데이터 전달이 대부분의 시간을 차지하므로, 이러한 데이터 전달 시간을 최소화하는 것이 중요하다.

Default 스트림만을 사용한 경우에는 data transfer와 kernel execution이 호출된 순서대로 device에서 수행된다.

반면 non-default 스트림들을 사용하여 호출하는 경우, 각 스트림에서의 data transfer와 kernel execution이 서로 오버랩되어 실행되는 것을 볼 수 있다. 초록색 `Memcpy`는 HtoD transfer를 나타내고 보라색 `Memcpy`는 DtoH transfer를 나타내는데, PCIe bus를 공유하기 때문에 각 스트림에서의 HtoD transfer와 DtoH transfer는 서로 오버랩되지 않는다는 것을 확인할 수 있다.

> 비동기 data transfer에서는 pinned host memory를 사용해야 한다는 점에 유의해야 한다. 만약 pinned host memory가 아닌 pageable host memory를 사용하면, 비동기 API이더라도 비동기로 동작하지 않을 수 있다. 실제로 [vector_add_with_streams.cu](/cuda/code/streams/vector_add_with_streams.cu)에서 pinned host memory가 아닌 pageable host memory를 사용한 결과, 약 16 ms이 걸리는 것으로 측정되었으며, `nsight system`으로 살펴보면 아래와 같이 비동기로 동작하지 않았다는 것을 확인할 수 있었다.
> 
> <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcrIRrD%2Fbtr1XynE6Ix%2FpSQgYgkmOcRMHU1jLbvTJ1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>
> 

> 동시에 실행할 수 있는 커널의 최대 갯수는 GPU device마다 다르다. CUDA 문서의 [Table 15](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)에서 `Maximum number of resident grids per device`에서 최대 갯수를 확인할 수 있으며, compute capability 7.5 이상의 GPU에서는 128개의 커널을 동시에 실행할 수 있다.

<br>

# Stream Scheduling

개념상, 모든 스트림은 동시에 동작할 수 있다. 하지만 물리적 하드웨어 관점에서 항상 동시에 동작하는 것은 아니다. 이번에는 하드웨어에서는 여러 CUDA 스트림의 concurrent kernel operation이 어떻게 스케줄링되는지 살펴보자.

## False Dependencies

Fermi 아키텍처의 경우 16개의 커널을 동시에 실행시킬 수 있도록 지원한다. 하지만 모든 스트림은 궁극적으로 하나의 hardware work queue에 들어오게 된다. 실행할 커널을 선택할 때, 이 큐에 가장 앞에 있는 작업은 CUDA 런타임에 의해 스케줄링된다. 런타임에서는 작업의 의존성을 체크하고 만약 다른 작업이 수행중이라면 그 작업이 완료될 때까지 해당 작업을 대기시킨다. 결과적으로 모든 의존성이 해결되어야 새로운 작업을 SM에 할당하여 실행시킨다. 따라서, 이러한 single pipeline(`single work queue`)는 **false dependency** 를 야기한다. 아래 그림은 이러한 false dependency를 잘 보여주고 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fm8wRa%2FbtrrpomD1J3%2FsKxUFhjreuVqSjf6qw7lf0%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

여러 스트림을 사용해서 여러 커널이나 연산들을 동시에 실행시키더라도 위 그림에서 빨간색 동그라미로 표시된 작업들만이 동시에 수행된다. 단일 hardware work queue에서 블록된 연산들은 서로 다른 스트림에 속해있더라도 해당 스트림의 후속 연산들을 모두 블록시킨다.

> 최근 GPU들은 아래에서 설명할 hyper-q를 포함하고 있기 때문에 false dependency를 확인할 수 없는 것으로 보인다.

## Hyper-Q

Kepler 아키텍처부터 false dependency를 줄이기 위해 **Hyper-Q** 라는 기술을 적용하여 여러 hardware work queue를 사용할 수 있다. Hyper-Q를 사용하면 host와 device 간 multiple hardware-managed connections을 유지하여 하나의 GPU에서 동시 작업이 가능하도록 해준다. Fermi 아키텍처에서는 false dependency로 인해 동시성이 많이 제한되어 있었지만, Kepler 아키텍처에서는 hyper-q 덕분에 기존의 코드 변경없이 성능 향상을 얻을 수 있다. Kepler GPU는 32개의 hardware work queues가 있으며 스트림당 하나의 queue를 할당하여 사용한다. 만약 32개 이상의 스트림이 사용되면 여러 스트림이 하나의 hardware work queue를 공유하게 된다. 아래 그림은 3개의 hardware work queue를 가진 장치에서 3개의 스트림을 사용하는 상황을 나타낸다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbkVvpq%2Fbtrroc8vyzi%2F4OdiIFwfZkDenn2e8svON1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

<br>

# Stream Priorities

Compute capability 3.5 이상에서부터 스트림에 우선순위를 할당할 수 있다. 우선순위는 상대적이며 `cudaStreamCreateWithPriority()` 함수를 통해 우선순위를 갖는 스트림을 생성할 수 있다.
```c++
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
```

> `flags` 파라미터에 대해서는 아래 [Stream Synchronization](#stream-synchronization)에서 설명

이 함수는 지정된 integer priority를 갖는 스트림을 생성하며, `pStream`에 스트림 핸들을 리턴한다. 이 우선순위는 `pStream` 내에서 스케줄링되는 작업들과 연관되어 있다. 우선순위가 높은 스트림에서 대기 중인 그리드는 우선 순위가 낮은 스트림에서 실행 중인 작업을 선점할 수 있다. 스트림의 우선순위는 data transfer 연산에는 영향을 주지 않으며, 오직 커널에만 영향을 미친다. 우선순위로 지정할 수 있는 정수값의 범위는 아래 API를 통해 쿼리할 수 있다.
```c++
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
```

따라서, 우선순위가 가장 높은 스트림과 가장 낮은 스트림을 생성하려면 다음과 같이 코드를 작성하면 된다.
```c++
// get the range of stream priorities for this device
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

범위를 벗어나는 값을 지정하면, 자동으로 가장 높거나 낮은 우선순위로 지정된다. 스트림 우선순위를 지원하지 않는 GPU에서는 두 파라미터 값은 0으로 리턴된다.

<br>

# Stream Synchronization

Non-default stream에서 실행되는 모든 연산들은 host 스레드에 대해 non-blocking, 즉, 비동기이다. 따라서, 스트림에서 수행 중인 연산과 host 간 동기화가 필요한 상황이 있을 수 있다.

Host 관점에서 CUDA 연산은 아래의 두 가지 카테고리로 분류될 수 있다.

- Memory-related operations
- Kernel launches

Kernel launches는 항상 host에 대해 비동기이다. 대부분의 메모리 연산(ex: `cudaMemcpy`)은 본질적으로 동기식(synchronous)이지만, CUDA 런타임 함수는 메모리 연산을 수행하는 비동기 함수도 제공한다(ex: `cudaMemcpyAsync`).

위에서 살펴봤듯, 스트림에는 두 가지 타입이 있다.

- Asynchronous streams (non-NULL(default) streams)
- Synchronous stream (NULL(default) stream)

Non-default stream은 host에 대해 비동기이므로, 여기서 적용되는 모든 연산들은 host를 블로킹하지 않는다. 반면, NULL-stream, 즉, default 스트림은 host에 대해 동기식이다. Default 스트림에 추가되는 대부분의 연산들은 host를 블로킹하여 해당 연산이 끝날 때까지 대기하도록 한다 (kernel launches 제외).

사실, non-default stream은 다음의 두 가지 타입으로 더 분류할 수 있다.

- Blocking streams
- Non-blocking streams

Non-default 스트림은 host에 대해 non-blocking이지만, non-default 스트림 내의 연산들은 default 스트림의 연산에 의해 블로킹될 수 있다. 이는 non-default 스트림을 어떤 `flag`로 생성하느냐에 따라 달라지는데, 만약 스트림을 **blocking stream** 으로 생성하면, default 스트림은 해당 스트림(non-default)에서의 연산을 블로킹할 수 있다. 반면, non-default 스트림을 **non-blocking stream** 으로 생성하면, default 스트림에서의 연산이 블로킹되지 않는다.

## Blocking and Non-Blocking Streams

`cudaStreamCreate`로 생성되는 스트림은 **blocking stream** 이며, 이는 default 스트림에 의해서 해당 스트림에서의 연산이 블로킹될 수 있다는 것을 의미한다.

Default 스트림은 동일한 CUDA context에서 다른 모든 blocking streams과 동기화되는 implicit stream이다. 일반적으로 default 스트림에 어떤 연산이 추가되면, CUDA context는 이 연산을 실행하기 전에 (이 연산이 추가되기 전에 추가된)모든 blocking 스트림에 추가되었던 모든 작업이 완료될 때까지 대기한다. 또한, 해당 연산이 추가되고 난 후, 다른 모든 blocking 스트림에서 추가된 연산들은 default 스트림의 연산이 완료될 때까지 대기한다.

CUDA 런타임은 non-default 스트림의 동작을 커스터마이즈할 수 있는 `cudaStreamCreateWithFlags()`를 제공한다.
```c++
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
```

`flags` 인자는 생성되는 스트림의 동작을 결정하며, 가능한 값은 아래와 같다.

- `cudaStreamDefault`(0) : default stream creation flag (blocking)
- `cudaStreamNonBlocking`(1) : asynchronous stream creation flag (non-blockgin)

예제 코드([blocking_and_non_blocking.cu](/cuda/code/streams/blocking_and_non_blocking.cu))를 통해 간단히 위 현상에 대해 살펴보자.

이 예제에서는 4개의 스트림을 생성할 때, flag(0 or 1)를 지정해준다. 0은 non-default 스트림을 blocking으로 생성하고, 1은 non-blocking으로 생성한다. 그런 다음, 아래 코드와 같이 `kernel_1`, `kernel_2`, `kernel_3`, `kernel_4`을 for문을 돌면서 순차적으로 네 번씩 실행한다. 이때, `kernel_3`은 무조건 default 스트림으로 실행하며, 나머지 커널들은 각 스트림에서 실행되도록 한다.
```c++
// launch kernels with streams
for (int i = 0; i < NUM_STREAMS; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>(num_elements);
    kernel_2<<<grid, block, 0, streams[i]>>>(num_elements);
    kernel_3<<<grid, block>>>(num_elements);
    kernel_4<<<grid, block, 0, streams[i]>>>(num_elements);
}
```

예제 코드를 컴파일하고 난 뒤, flag를 0(blocking stream)으로 지정하고 `nsight systems`으로 실행해보면 아래와 같이 커널들이 실행된 것을 확인할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbogdtl%2Fbtr133UK3sl%2F4xgTiJrVGaP6FRlaX36FNk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

보기 좋게 각 커널들이 동시에 실행되지는 않았지만, `kernel_4`와 `kernel_1`, `kernel_2`가 서로 오버랩되어 있는 것은 확인할 수 있다. 하지만, default 스트림에서 실행되는 `kernel_3`과는 오버랩되는 것을 볼 수 없다. 커널을 실행할 때, for문을 순회하면서 `kernel_1`부터 `kernel_4`까지 순차적으로 호출된다. 하지만 `kernel_3`은 default 스트림에서 호출되므로, `kernel_3`이 실행되기 전에 이전에 호출되었던 `kernel_1`과 `kernel_2`가 완료될 때까지 default 스트림은 대기하게 된다. `kernel_1`과 `kernel_2`의 수행이 완료되면 그제서야 `kernel_3`이 default 스트림에서 실행되며, 이후에 다른 non-default 스트림에서 호출된 커널들은 `kernel_3`이 완료될 때까지 블로킹된다.

> Non-default 스트림을 살펴보면, 먼저 호출된 `kernel_4` 커널의 실행이 이후에 호출된 다른 스트림의 `kernel_1`보다 늦게 실행되는 것을 관찰할 수 있다. 이는 non-default 스트림 간의 실행 순서는 호출된 순서가 아니라는 것을 보여준다.

반면, flag를 1(non-blocking stream)로 지정하여 `nsight systems`를 실행한 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbLfBdf%2Fbtr1VeXwzsi%2FNvZu7fUb6d4ZA0hRA9wPGK%2Fimg.png" width=800px style="display: block; margin: 0 auto"/>

Non-default 스트림을 non-blocking으로 지정하면 default 스트림에 더 이상 블로킹되지 않는다. 따라서, 더 이상 default 스트림에서 실행되는 `kernel_3`이 다른 커널 실행들을 블로킹하지 않아 랜덤한 순서로 실행되는 것을 볼 수 있다.

> 예제의 전체 코드는 [blocking_and_non_blocking.cu](/cuda/code/streams/blocking_and_non_blocking.cu)를 참조

## Implicit Syncrhonization

CUDA의 host-device간 동기화에는 `explicit`과 `implicit` 동기화가 있다. `cudaDeviceSynchronize()`과 `cudaStreamSynchronize(stream)` 런타임 API 함수는 explicit 동기화를 수행한다. 이 함수들은 host 측에서 호출하여 device의 작업과 host 스레드 간 동기화를 수행한다.

반면 `cudaMemcpy`와 같은 함수는 host와 device 간의 implicit 동기화를 수행하는데, 이 함수는 data transfer가 완료될 때까지 host를 블로킹하기 때문이다. 이 함수의 주 목적이 동기화를 하는 것이 아니므로, 이러한 함수를 실수로 많이 호출하게 되면 예상치 못한 성능 저하가 발생하므로 어떤 함수들이 내부적으로 동기화를 수행하는지 알아두는 것이 좋다. 

서로 다른 스트림의 두 연산 사이에 아래의 연산이 host 측에서 수행되는 경우, 각 스트림의 연산은 동시에 실행될 수 없다. 예를 들어, host 측에서 순차적으로 1번 스트림에서 A 연산을 실행하고 2번 스트림에서 B 연산을 수행하는 상황을 가정해보자. 만약 A 연산과 B 연산 사이에 아래의 연산 중 하나가 실행된다면, A 연산과 B 연산은 서로 다른 스트림에서 실행되더라도 동시에 수행될 수 없다.

- a page-locked host memory allocation
- a device memory allocation
- a device memory set(memset)
- a memory copy between two addressed to the same device memory
- any CUDA command to the NULL(default) stream
- a switch between the L1/shared memory configuration

## Explicit Synchronization

[Implicit Syncrhonization](#implicit-syncrhonization)에서도 언급했지만 CUDA 런타임에는 grid level에서 명시적인 동기화를 수행하는 몇 가지 방법을 지원한다.

- Synchronizing the device
- Synchronizing a stream
- Synchronizing an event in a stream
- Synchronizing across streams using an event

> CUDA Event에 대해서는 아직 살펴보진 않았다. 간단히 언급하자면, CUDA event는 프로그램의 어느 시점에서의 이벤트를 비동기식으로 기록(record)하고, 이러한 event가 완료되는 시점을 쿼리하여 device에서의 진행 상황을 모니터링하고 정확한 타이밍을 맞추는 방법을 제공한다.

먼저 device에서 이전에 실행된 모든 작업들이 완료될 때까지 host 스레드를 블로킹하려면 아래 함수를 호출하면 된다.

```c++
cudaError_t cudaDeviceSynchronize(void);
```

이 함수는 현재 device와 연관된 모든 연산과 통신이 완료할 때까지 host 스레드를 기다리도록 한다. 상당히 성능에 영향을 많이 주는 함수이므로 자주 사용하는 것은 좋지 않다.

어느 스트림에서 실행된 모든 연산이 완료될 때까지 host 스레드를 블로킹하려면 아래 함수를 사용한다.
```c++
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```

또는, non-blocking으로 스트림의 모든 연산이 완료되었는지 확인하려면 아래 함수를 사용한다.
```c++
cudaError_t cudaStreamQuery(cudaStream_t stream);
```

위의 두 함수에 대한 설명은 [CUDA Streams](#cuda-streams)에서 확인할 수 있다.

<br>

> **Non-NULL 스트림** 에서의 자세한 동작 및 예제 코드는 [Concurrent Kernel Execution](/cuda/study/14-1_concurrent_kernel_execution.md)에서 이어서 다루니, 이를 참조 바람.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- [NVIDIA CUDA Documentation: Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization)
