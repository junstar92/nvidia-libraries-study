# Table of Contents

- [Table of Contents](#table-of-contents)
- [Stream Callbacks](#stream-callbacks)
- [References](#references)

<br>

# Stream Callbacks

스트림 콜백(stream callback)은 CUDA 스트림 queue에서 수행되는 또 다른 타입의 연산이다. 스트림 콜백 이전에 실행된 스트림의 모든 작업들이 완료되면, 스트림 콜백에 의해 지정된 host 함수가 CUDA 런타임에 의해 호출된다. 호출되는 host 함수는 프로그램 내에서 정의 및 구현해야 한다.

스트림 콜백은 CPU-GPU 간 동기화를 수행하는 또 다른 메커니즘이다. 스트림 콜백 함수는 프로그램 내에서 직접 구현해야 하는 host 함수이며, 아래 CUDA 런타임 API를 통해 스트림에 등록한다.

```c++
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

> `cudaStreamAddCallback`은 deprecated and removal 예정이다. 대신 `cudaLaunchHostFunc`을 사용할 수 있다. `cudaLaunchHostFunc`에 대해서는 아래에서 다룬다.

이 함수는 콜백 함수를 주어진 스트림에 추가한다. 추가된 콜백 함수는 이전에 스트림 대기열에 있는 모든 작업들이 완료된 후 호출된다. `cudaStreamAddCallback` 당 한 번만 콜백 함수가 실행되며, 콜백 함수가 완료될 때까지 다른 대기 중인 작업들을 블로킹한다. `cudaStreamAddCallback`에 `userData` 파라미터를 사용하여 콜백 함수에 전달될 프로그램의 데이터를 지정할 수도 있다. `flags` 파라미터는 reserved이며, 현재는 무조건 0으로 지정해야 한다.

콜백 함수에는 다음의 두 가지 제약 조건이 있다.

- CUDA API 함수를 콜백 함수로 등록할 수 없다
- 콜백 함수 내에서 동기화를 수행할 수 없다

일반적으로 서로 다른 콜백 함수 또는 다른 CUDA 연산과 콜백 함수의 순서를 가정하는 것은 위험하며, 불안정한 코드가 될 수 있다.

`cudaStreamAddCallback`에 전달할 콜백 함수는 아래와 같이 정의 및 구현할 수 있다.
```c++
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("Callback from stream %d\n", *((int*)data));
}
```

그리고 다음과 같이 각 스트림에 콜백 함수를 추가할 수 있다.
```c++
for (int i = 0; i < num_streams; i++) {
    stream_ids[i] = i;
    kernel_1<<<grid, block, 0, streams[i]>>>(i);
    kernel_2<<<grid, block, 0, streams[i]>>>(i);
    kernel_3<<<grid, block, 0, streams[i]>>>(i);
    kernel_4<<<grid, block, 0, streams[i]>>>(i);
    cudaStreamAddCallback(streams[i], myCallback, &stream_ids[i], 0);
}
```

> 전체 코드는 [stream_callbacks.cu](/cuda/code/streams/stream_callbacks.cu)을 참조

위 코드를 컴파일하고, 실행하면 아래와 같은 출력을 얻을 수 있다. 콜백 함수가 출력되는 시점을 실행할 때마다 다르다.
```
$ ./stream_callbacks
> At device 0: NVIDIA GeForce RTX 3080 with num_streams=4
> Compute Capability 8.6 hardware with 68 multi-processors
[stream 0] kernel_1
[stream 0] kernel_2
[stream 0] kernel_3
[stream 1] kernel_1
[stream 2] kernel_1
[stream 3] kernel_1
[stream 0] kernel_4
[stream 1] kernel_2
[stream 2] kernel_2
[stream 3] kernel_2
[stream 1] kernel_3
[stream 2] kernel_3
[stream 3] kernel_3
Callback from stream 0
[stream 1] kernel_4
[stream 2] kernel_4
Callback from stream 1
[stream 3] kernel_4
Callback from stream 2
Callback from stream 3
Measured time for parallel execution: 1.198 ms
```

<br>

`cudaStreamAddCallback`은 deprecated 예정이고, API description을 보면 `cudaLaunchHostFunc`을 사용하라고 언급하고 있다.
```c++
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
```

사용 방법은 기존과 동일하며, `flags` 파라미터가 없어졌다.

위 함수를 통해 콜백 함수를 추가할 때, 콜백 함수 정의가 기존과 조금 다르다. 기존에는 `cudaStream_t`와 `cudaError_t`를 파라미터로 받는 콜백 함수를 정의했어야 했는데, `cudaLaunchHostFunc`으로 전달하는 콜백 함수는 사용자 데이터를 전달받기 위한 `void*` 파라미터만 받으면 된다. 따라서, 기존의 `myCallback` 함수는 아래와 같이 수정되어야 한다.
```c++
void CUDART_CB myCallback(void* data)
{
    printf("Callback from stream %d\n", *((int*)data));
}
```

그리고, 아래 코드에서와 같이 `cudaLaunchHostFunc`을 사용해서 콜백 함수를 추가하면 된다.
```c++
for (int i = 0; i < num_streams; i++) {
    stream_ids[i] = i;
    kernel_1<<<grid, block, 0, streams[i]>>>(i);
    kernel_2<<<grid, block, 0, streams[i]>>>(i);
    kernel_3<<<grid, block, 0, streams[i]>>>(i);
    kernel_4<<<grid, block, 0, streams[i]>>>(i);
    cudaLaunchHostFunc(streams[i], myCallback, &stream_ids[i]);
}
```

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher