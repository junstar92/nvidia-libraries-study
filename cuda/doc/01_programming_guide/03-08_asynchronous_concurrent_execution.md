# Table of Contents

- [Table of Contents](#table-of-contents)
- [Asynchronous Concurrent Execution](#asynchronous-concurrent-execution)
  - [Concurrent Execution between Host and Device](#concurrent-execution-between-host-and-device)
  - [Concurrent Kernel Execution](#concurrent-kernel-execution)
  - [Overlap of Data Transfer and Kernel Execution](#overlap-of-data-transfer-and-kernel-execution)
  - [Concurrent Data Transfers](#concurrent-data-transfers)
- [Streams](#streams)
  - [Creation and Destruction](#creation-and-destruction)
  - [Default Stream](#default-stream)
  - [Explicit Synchronization](#explicit-synchronization)
  - [Implicit Synchronization](#implicit-synchronization)
  - [Overlapping Behavior](#overlapping-behavior)
  - [Host Functions (Callbacks)](#host-functions-callbacks)
  - [Stream Priorities](#stream-priorities)
- [References](#references)

<br>

# Asynchronous Concurrent Execution

CUDA에서는 아래의 연산들을 서로 독립적으로 동시에 수행할 수 있다.

- Computation on the host
- Computation on the device
- Memory transfers from the host to the device
- Memory transfers from the device to the host
- Memory transfers within the memory of a given device
- Memory transfers among devices

위 연산들 간에 달성할 수 있는 `the level of concurrency`는 GPU device의 feature set과 compute capability에 따라 다르다.

## Concurrent Execution between Host and Device

GPU device와 동시에 host를 실행하는 것은 host에서 요청된 작업을 device가 완료하기 전에 제어권을 바로 host 스레드로 반환하는 비동기 API를 통해 가능하다. 아래의 device 연산들은 host에 대해 비동기로 동작한다 (즉, 호출되자마자 바로 host 스레드로 제어권을 반환한다).

- Kernel launches
- Memory copies within a single device's memory
- Memory copies from host to device of a memory block of 64KB or less
- Memory copies performed by functions that are suffixed with `Async`
- Memory set function calls

> `CUDA_LAUNCH_BLOCKING`이라는 환경 변수를 `1`로 설정하면 시스템에서 실행 중인 모든 CUDA application에 대해 비동기 동작을 비활성화할 수 있다. 이는 디버깅 목적으로만 사용해야 한다.

> 만약 hardware counter가 프로파일러(`nsight`, `visual profiler`)에 의해 수집되는 경우, kernel launches가 동기식이다. 또한, `page-locked` host memory가 아닌 메모리의 복사도 동기식으로 동작한다.

## Concurrent Kernel Execution

최근 대부분의 GPU device에서는 여러 커널을 동시에 실행할 수 있다. 동시에 커널을 실행할 수 있는지 여부는 `device property`를 쿼리하여 `concurrentKernels` 변수를 통해 확인할 수 있다. `concurrentKernels`가 1이라면 해당 device는 여러 커널을 동시에 실행할 수 있다.
```c++
int dev = 0;
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, dev);
printf("Device can execute multiple kernels concurrently: %s\n", prop.concurrentKernels ? "yes" : "no");
```

Compute capability에 따라 동시에 실행할 수 있는 커널의 최대 갯수는 다른데, 이는 NVIDIA CUDA 문서의 [Table 15](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)의 `Maximum number of resident grids per device` 행에서 확인할 수 있다.

> 한 CUDA context의 커널은 다른 CUDA context의 커널과 동시에 실행될 수 없다.

## Overlap of Data Transfer and Kernel Execution

GPU device에서 커널과 비동기 메모리 복사(`to or from GPU`)를 동시에 수행할 수도 있다. 이 기능은 `cudaDeviceProp`의 `asyncEngineCount`를 쿼리하여 확인할 수 있다.
```c++
printf("  Concurrent copy and kernel execution: %s with %d copy engine(s)\n",
    (dev_prop.deviceOverlap ? "Yes" : "No"), dev_prop.asyncEngineCount);
```
만약 비동기 메모리 복사에 host memory가 연관되어 있다면, 이 host memory는 **page-locked** 상태이어야만 한다.

또한, kernel execution 과(또는) copies to or from device와 동시에 intra-device copy를 수행할 수도 있다.

## Concurrent Data Transfers

최근 device에서는 `host to device`와 `device to memory` 메모리 복사 연산을 동시에 수행할 수 있으며, 이 기능은 위와 같이 `cudaDeviceProp`의 `asyncEngineCount`를 쿼리하여 가능한지 확인할 수 있다. 이 값이 `2`라면 `host to device`와 `device to memory` 메모리 복사 연산을 동시에 수행할 수 있다. 단, host memory는 반드시 **page-locked** 이어야만 한다.

<br>

# Streams

지금까지 위에서 설명한 동시 작업들은 **스트림(streams)** 을 통해 관리한다. 스트림은 순서대로 실행되는 일련의 커맨드이며, 서로 다른 스트림은 순서와 상관없이 동시에 커맨드를 실행할 수 있다. 서로 다른 스트림 간의 순서는 결정적이지 않다.

스트림에 대한 기본적인 내용은 아래 포스팅에서 자세히 다루고 있다.

- [Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md)
- [Concurrent Kernel Execution](/cuda/study/14-1_concurrent_kernel_execution.md)
- [Overlapping Kernel Execution and Data Transfer](/cuda/study/14-2_overlapping_kernel_execution_and_data_transfer.md)
- [Stream Callback](/cuda/study/14-3_stream_callback.md)

## Creation and Destruction

스트림은 스트림 객체를 생성해 정의한다. 그리고, 커널을 호출하거나 메모리 복사할 때 스트림 파라미터에 스트림을 전달하여 사용할 수 있다. 아래 예제 코드는 두 개의 스트림을 생성하고, `hostPtr` 배열을 `page-locked` memory로 할당한다.
```c++
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

이렇게 생성한 스트림은 아래 코드에서와 같이 사용할 수 있다.
```c++
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

스트림을 사용하는 자세한 방법은 위에 나열한 스트림 관련 포스팅에서 다루고 있으니 필요하면 이를 참조하면 된다.

사용이 끝난 스트림은 `cudaStreamDestroy()`를 호출하여 해제할 수 있다.
```c++
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

만약 `cudaStreamDestroy()`가 호출될 때 스트림에서 여전히 작업을 수행하고 있다면, 이 함수는 즉시 반환되며 스트림에서 모든 작업이 완료될 때 스트림과 관련된 리소스가 자동으로 해제된다.

## Default Stream

커널을 호출할 때나 host<->device간 메모리 복사할 때 스트림 파라미터를 지정하지 않거나 `0(NULL)`로 지정하면 **default 스트림**에서 실행된다. 따라서, 각 연산들은 순서대로 실행된다.

`--default-stream per-thread` 컴파일 옵션으로 컴파일된 코드나 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 매크로가 CUDA 헤더 파일(`cuda.h` and `cuda_runtime.h`)을 include하기 전에 정의되면, default 스트림은 regular stream이고 각 host 스레드에서 저마다의 default 스트림을 가지게 된다.

> `nvcc`는 암시적으로 TC의 첫 부분에서 `cuda_runtime.h`를 include하기 때문에, `nvcc`로 코드를 컴파일할 때 `#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1`을 통해 방금 설명한 동작을 활성화시킬 수 없다. 이 경우에는 `--default-stream per-thread` 컴파일 플래그를 설정하거나, `-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1`와 같이 컴파일 플래그를 통해 매크로를 정의해야 한다.

`--default-stream legacy` 컴파일 옵션으로 컴파일된 코드에서 default 스트림은 `NULL` 스트림이라 부르는 special stream이다. 이때, 각 device는 하나의 NULL 스트림을 가지며 이는 모든 host 스레드에서 사용된다. NULL stream은 [Implicit Synchronization](#implicit-synchronization)에서 언급하는 **implicit synchronization**을 발생시키기 때문에 특별하다.

`--default-stream` 옵션을 지정하지 않으면 기본으로 `--default-stream legacy`으로 설정된다 (우리가 일반적으로 사용하는 경우에 해당).

## Explicit Synchronization

아래의 API를 통해 명시적으로 스트림에 동기화할 수 있다.

- `cudaDeviceSynchronize()` : 모든 host 스레드의 모든 스트림의 이전 작업들이 완료될 때까지 기다린다
- `cudaStreamSynchronize()` : 이 API는 하나의 스트림을 인자로 받으며, 이 스트림의 이전 작업들이 완료될 때까지 기다린다 (host와 특정 스트림을 동기화시키고, 다른 스트림은 계속 작업을 수행하도록 하는데 사용됨)
- `cudaStreamWaitEvent()` : 스트림과 이벤트를 인자로 받으며, 이 API가 호출된 이후 추가된 모든 작업들을 주어진 이벤트가 완료될 때까지 실행을 지연시킨다
- `cudaStreamQuery()` : 스트림을 인자로 받으며, API가 호출된 지점에서 스트림에 있는 모든 작업들이 완료되었는지 체크하고, 그 여부를 리턴한다

## Implicit Synchronization

아래의 작업들 중 하나가 host 스레드에서 서로 다른 스트림의 두 연산 사이에 실행되는 경우, 해당 스트림들의 연산은 동시에 실행될 수 없다. 따라서, 아래 작업들 중 하나가 host 스레드에서 실행되면, 명시적으로 동기화시키지 않았지만 이 작업이 완료될 때까지 이후에 실행된 다른 스트림에서의 작업들이 대기한다.

- a page-locked host memory allocation
- a device memory allocation
- a device memory set
- a memory copy between two addresses to the same device memory
- any CUDA command to the NULL stream
- a switch between the L1/shared memory configurations <br> ([NVIDIA CUDA 문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x) 또는 [Shared Memory: Configuring the Amount Of Shared Memory](/cuda/study/12_shared_memory.md#configuring-the-amount-of-shared-memory) 참조)

## Overlapping Behavior

두 스트림 간의 연산을 동시에 실행할 수 있는지는 device의 `overlap of data transfor and kernel execution`, `concurrent kernel execution`, `concurrent data transfers` 지원 여부에 따라 다르다.

Device에서 concurrent data transfers를 지원한다면, 아래 예제 코드처럼 실행하면 kernel execution과 data transfer(DtoH, HtoD), 그리고 서로 다른 스레드의 DtoH와 HtoD memory copy를 동시에 실행할 수 있다.
```c++
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

하지만, device에서 concurrent data transfers를 지원하지 않는다면 `false dependencies`로 인해 위의 코드에서 오버랩은 불가능하다. 이 경우에는 `streams[1]`에서의 DtoH memory copy가 `streams[0]`의 HtoD memory copy 이후에 호출된다. 따라서, `streams[0]`의 DtoH memory copy가 완료되어야만 `streams[1]`의 HtoD memory copy가 실행될 수 있다. 하지만, 위의 코드를 아래와 같이 수정하면 `false dependencies` 문제를 해결하여, `streams[1]`의 HtoD memory copy와 `streams[0]`의 kernel launch를 오버랩시킬 수 있다.
```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

> `False Dependencies`에 관한 내용은 [Introducing CUDA Streams: False Dependencies](/cuda/study/14_introducing_cuda_streams.md#false-dependencies)를 참조

## Host Functions (Callbacks)

런타임 API에서 `cudaLaunchHostFunc()`를 통해 특정 시점에 CPU 함수(callback) 호출을 스트림에 추가하는 방법을 제공한다.

아래 예제 코드는 `MyCallback`이라는 host 함수를 두 스트림에 추가한다. 콜백 함수는 동일한 스트림에서 먼저 추가된 HtoD memory copy, kernel launch, DtoH memory copy가 모두 완료된 이후에 실행된다.
```c++
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)i);
}
```

콜백 함수가 스트림에 추가된 이후에 해당 스트림에서 호출되는 작업들은 콜백 함수가 완료되기 전에 실행되지 않는다.

> Note: 스트림 큐에 추가된 host 콜백 함수에서는 CUDA API를 호출해서는 안된다 (Dealock을 유발할 수 있음).

## Stream Priorities

`cudaStreamCreateWithPriority()`를 통해 스트림을 생성하면, 스트림 간의 상대적인 우선순위를 지정할 수 있다. 지정할 수 있는 우선순위의 범위는 `cudaDeviceGetStreamPriorityRange()`를 통해 얻을 수 있으며, 그 범위는 [highest priority, lowest priority] 이다. 런타임에서 우선 순위가 높은 스트림에서의 작업이 우선 순위가 낮은 스트림에서의 작업보다 우선시된다.

아래 예제 코드는 현재 device에서 사용 가능한 우선순위의 범위를 얻은 뒤, 가장 높은 우선순위와 가장 낮은 우선순위로 두 개의 스트림을 생성하는 방법을 보여준다.
```c++
// get the range of stream priorities for this device
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

<br>

# References

- [NVIDIA CUDA Documentations: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- [NVIDIA CUDA Sample Codes](https://github.com/NVIDIA/cuda-samples)
  - [simpleStreams](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleStreams) - overlap kernel execution with memory copies
  - [concurrentKernels](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels) - concurrent execution by using streams, depedencies between CUDA streams with `cudaStreamWaitEvent`
  - [simpleHyperQ](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleHyperQ) - demonstrates HyperQ
  - [simpleCallback](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleCallback) - host callback function in the streams