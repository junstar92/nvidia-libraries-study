# Table of Contents

- [Table of Contents](#table-of-contents)
- [CUDA Events](#cuda-events)
  - [Creation and Destruction](#creation-and-destruction)
  - [Recording Events and Measuring Elapsed Time](#recording-events-and-measuring-elapsed-time)
- [Creating Inter-Stream Dependencies](#creating-inter-stream-dependencies)
- [Overlapping GPU and CPU Execution](#overlapping-gpu-and-cpu-execution)
- [References](#references)

# CUDA Events

CUDA **Event** 는 CUDA 스트림 내에서 마커 역할을 수행한다. 즉, 스트림 내 어떤 연산의 어떤 특정 시점을 비동기식으로 기록(record)한다. 그리고 이렇게 기록한 이벤트가 완료되었는지 쿼리할 수 있으며 device의 진행 현황을 모니터링하고 정확한 타이밍을 위한 방법을 제공한다. 이벤트가 기록된 이전의 연산들이 완료되면 해당 이벤트는 완료된 것이다. Default 스트림 내에서의 이벤트는 모든 스트림의 선행 작업과 커맨드가 끝나야 완료된다.

CUDA 이벤트를 사용하면 기본적으로 아래의 두 작업을 수행할 수 있다.

- Synchronize stream execution
- Monitor device progress
- Time the code

## Creation and Destruction

CUDA 이벤트를 아래와 같이 선언할 수 있고,
```c++
cudaEvent_t event;
```

아래의 CUDA 런타임 API를 사용하여 이벤트를 생성할 수 있다.
```c++
cudaError_t cudaEventCreate(cudaEvent_t* event);
```

그리고, 아래 API를 사용하여 이벤트 리소스를 해제할 수 있다.
```c++
cudaError_t cudaEventDestroy(cudaEvent_t event);
```

아래의 코드는 두 개의 이벤트를 생성 및 해제하는 것을 보여준다.
```c++
cudaEvent_t start, end;
cudaEventCreate(&start);
cudaEventCreate(&end);
...
cudaEventDestroy(start);
cudaEventDestroy(end);
```

만약 `cudaEventDestroy`가 호출될 때, 이벤트가 아직 완료되지 않았다면 `cudaEventDestroy`는 즉시 리턴되며 이벤트가 완료될 때 자동으로 해당 이벤트가 해제된다.

## Recording Events and Measuring Elapsed Time

CUDA 이벤트는 스트림 실행 내의 특정 지점을 마킹한다. 그리고, 이는 실행 중인 스트림 연산이 해당 지점에 도착했는지 체크하는데 사용될 수 있다. 이벤트는 스트림에 추가된 연산으로 생각할 수 있으며, 아래 API를 사용하여 스트림에 이벤트를 추가할 수 있다.

```c++
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
```

API를 통해 전달된 이벤트는 지정된 스트림 내에서 선행 작업들의 완료를 기다리거나 테스트하는 데 사용할 수 있으며, 아래 API를 통해 host 스레드가 해당 이벤트가 완료될 때까지 대기하도록 할 수 있다.
```c++
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```

`cudaEventSynchronize()`는 `cudaStreamSynchronize()`와 유사하면서도, 스트림 내의 중간 지점을 기다리도록 할 수 있다는 차이점이 있다.

Host 스레드를 블로킹하지 않고 어떤 이벤트가 완료되었는지 확인하려면 아래 API를 사용하면 된다. 이 함수는 `cudaStreamQuery`와 거의 동일하다.
```c++
cudaError_t cudaEventQuery(cudaEvent_t event);
```

두 개의 CUDA 이벤트를 사용하면, 아래의 API를 통해 해당 이벤트 지점 사이의 연산 시간을 측정할 수 있다.
```c++
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
```
이 함수는 `start` 이벤트와 `stop` 이벤트 사이의 수행 시간을 msec 단위로 측정한다. `start`와 `stop` 이벤트는 동일한 스트림 내에서의 이벤트일 필요는 없다.

두 이벤트 사이의 실행 시간을 측정하는 방법은 다음과 같다.
```c++
// create two events
cudaEvent_t start, end;
cudaEventCreate(&start);
cudaEventCreate(&end);

// record start event on the default stream
cudaEventRecord(start);

// execute kernel
kernel<<<grid, block>>>(...);

// record stop event on the default stream
cudaEventRecord(end);

// wait until the stop event completes
cudaEventSynchronize(end);

// calculate the elapsed time between start and end events
float msec;
cudaEventElapsedTime(&msec, start, stop);

// clean up the two events
cudaEventDestroy(start);
cudaEventDestroy(end);
```

> 스트림을 지정하지 않으면 기본적으로 default 스트림에 이벤트를 추가한다.

<br>

# Creating Inter-Stream Dependencies

이상적으로는 스트림 간의 의도하지 않은 종속성은 없어야 한다. 하지만 복잡한 프로그램에서는 다른 스트림에서의 작업이 어떤 스트림이 수행되기 전에 완료될 필요가 있거나 그 결과가 필요할 때가 있다. 이러한 경우에는 스트림 간 종속성을 추가하는 것이 유용할 수 있다.

만약 한 스트림에서의 작업들이 다른 모든 스트림의 작업이 완료된 이후에 실행되기를 원한다고 가정해보자. 즉, 스트림 간 종속성이 필요한 경우에 해당한다. CUDA 이벤트를 사용하면 스트림 간의 종속성을 생성할 수 있다.

> 스트림 간 종속성만 필요한 경우, 성능상의 이유로 `cudaEventCreateWithFlags`를 사용하는 것이 좋다. 이 API는 두 번째 인자로 `flags`를 받는데, 여기에 `cudaEventDisableTiming`를 전달하면 해당 이벤트는 수행 시간을 측정하지 않게 된다.

이를 위한 코드 구현은 다음과 같다. 먼저, `cudaEventCreateWithFlags`를 사용하여 동기화를 위한 이벤트를 다음과 같이 생성한다. 앞서 언급했듯이 스트림 간의 종속성만이 필요한 경우, 굳이 시간을 측정하는데 리소스를 사용할 필요가 없으므로 시간 측정 기능을 비활성화하기 위해 `cudaEventCreateWithFlags`로 이벤트를 생성한다. 당연히 `cudaEventCreate`로 생성한 이벤트도 사용 가능하다.
```c++
cudaEvent_t *kernel_events = (cudaEvent_t*)malloc(num_streams * sizeof(cudaEvent_t));
for (int i = 0; i < num_streams; i++) {
    cudaEventCreateWithFlags(&kernel_events[i], cudaEventDisableTiming);
}
```

그런 다음, `cudaEventRecord`를 사용하여 각 스트림의 작업들이 완료되는 지점에 각 이벤트를 기록한다. 그리고, `cudaStreamWaitEvent`를 사용하여 마지막 스트림은 다른 모든 스트림이 완료될 때까지 기다리도록 해준다.
```c++
for (int i = 0; i < num_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();

    cudaEventRecord(kernel_events[i], streams[i]);
    cudaStreamWaitEvent(streams[num_streams - 1], kernel_events[i], 0);
}
```

> 전체 코드는 [simple_event.cu](/cuda/code/events/simple_event.cu)를 참조

전체 코드를 컴파일하고 `nsight systems`로 프로파일링하여 시각화한 결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F0Bo1Z%2Fbtr2Gs1Xgsj%2FEWzFEEbSsNxvT7hN4WgBk0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

스트림 13,14,15의 작업들이 모두 완료된 이후에 마지막 스트림 16의 작업이 시작되는 것을 확인할 수 있다.

<br>

# Overlapping GPU and CPU Execution

모든 커널은 비동기로 동작하므로 GPU와 CPU에서 동시에 작업을 수행하도록 하는 것은 비교적 간단하다. 단순히 커널을 실행시키면, 그 즉시 제어권이 CPU 스레드로 반환되므로 CPU에서 필요한 작업을 수행하면 자동으로 GPU와 CPU에서 작업을 오버랩할 수 있다.

예제 코드를 통해 살펴보도록 하자.

> 전체 코드는 [async_api.cu](/cuda/code/events/async_api.cu)를 참조

예제 코드의 실행 과정은 두 가지 스테이지로 나눌 수 있다. 첫 번째는 default 스트림에서 kernel 및 memory copies를 실행한다. 두 번째 부분에서는 host 스레드가 GPU의 작업을 기다리면서 특정 연산을 수행한다. 예제 코드에서 CPU 연산은 단순히 카운터를 증가시키는 작업으로 구현했다.

예제 코드에서는 먼저 3가지 GPU 연산 (2 copies and kernel launch)을 실행한다. 그리고 `stop` 이벤트는 GPU의 모든 연산이 완료되는 지점에 마킹하여 기록하게 된다.
```c++
// asynchronous issue work to the GPU (all to default stream)
cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_data, val);
cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
cudaEventRecord(stop);
```

위의 3가지 GPU 연산(+`cudaEventRecord`)은 모두 host에 대해 비동기식으로 실행된다. 따라서, CUDA 런타임 API와 kernel launch는 호출되자마자 host 측으로 제어권을 반환하게 된다. 이후에 host 스레드는 아래의 코드를 곧바로 수행한다.
```c++
// have CPU do some work while waiting for GPU works to finish
unsigned long int counter = 0;
while (cudaEventQuery(stop) == cudaErrorNotReady) {
    counter++;
}
```

위 코드에서 host는 CUDA 이벤트를 통해 모든 작업이 완료되었는지 `while`문을 반복하면서 체크하며, 아직 완료되지 않았다면 `counter`의 값을 1 증가시킨다. `cudaEventQuery`를 통해 해당 이벤트가 완료되었는지 체크할 수 있으며, 아직 완료되지 않았다면 `cudaErrorNotReady`를 반환한다.

예제 코드를 실행한 결과는 다음과 같다.
```
$ ./async_api
> At device 0: NVIDIA GeForce RTX 3080
CPU executed 85347 iterations while waiting for GPU to finish
```

GPU 작업이 모두 완료될 때까지 host는 `while`문을 85,347번 반복했다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)