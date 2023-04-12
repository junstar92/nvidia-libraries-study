# Table of Contents

- [Table of Contents](#table-of-contents)
- [Performance Metrics](#performance-metrics)
- [Timing](#timing)
  - [Using CPU Timers](#using-cpu-timers)
  - [Using CUDA GPU Timers](#using-cuda-gpu-timers)
- [Bandwidth](#bandwidth)
  - [Theoretical Bandwidth Calculation](#theoretical-bandwidth-calculation)
  - [Effective Bandwidth Calculation](#effective-bandwidth-calculation)
  - [Throughput Reported by Visual Profiler](#throughput-reported-by-visual-profiler)
- [References](#references)

<br>

# Performance Metrics

CUDA 코드를 최적할 때, 성능을 정확하게 측정하는 방법과 성능 측정에서 bandwidth의 역할을 이해하는 것이 중요하다. 이번 포스팅에서는 CUDA의 타이머와 CUDA Events를 사용하여 성능을 올바르게 측정하는 방법에 대해서 다룬다. 또한, bandwidth가 성능 메트릭에 미치는 영향과 이와 관련한 몇 가지 문제점을 해결하는 방법에 대해 살펴본다.

<br>

# Timing

CUDA call과 kernel execution은 CPU 또는 GPU 타이머를 사용하여 시간을 측정할 수 있다. 두 가지 방법의 기능, 장점 및 놓치기 쉬운 함정에 대해서 살펴보자.

## Using CPU Timers

모든 CPU 타이머는 CUDA call과 kernel execution의 수행 시간을 측정할 수 있다.

CPU 타이머를 사용할 때는 많은 CUDA API 함수가 비동기로 동작한다는 것을 기억해야 한다. 즉, 작업을 완료하기 전에 CPU 스레드로 제어권을 반환한다. 함수 이름에 `Async` 접미사가 붙은 메모리 복사와 같은 함수와 마찬가지로 모든 커널 launch도 비동기식으로 동작한다. 따라서, 특정 call 또는 CUDA call 시퀀스에 대한 경과 시간을 정확하게 측정하려면 CPU 타이머를 시작/중지하기 직전에 `cudaDeviceSynchronize()`를 호출하여 CPU 스레드와 GPU를 동기화시켜 주어야 한다.

CPU 스레드와 GPU의 특정 스트림이나 이벤트와 동기화시키는 것도 가능하지만, 이는 default 스트림이 아닌 명시적 스트림에서 시간을 측정하는 코드에서는 적합하지 않다. `cudaStreamSynchronize()`는 주어진 스트림에서 이전에 실행된 모든 CUDA call이 완료될 떄까지 CUDA 스레드를 블로킹한다. `cudaEventSynchronize()`는 주어진 특정 스트림의 이벤트가 GPU에 의해 레코딩될 때까지 블로킹한다. Driver는 default가 아닌 다른 스트림의 CUDA call이 중간에 끼어들 수 있기 때문에, 다른 스트림의 call이 시간 측정에 포함될 수 있다.

> CPU-GPU 동기화는 GPU 파이프라인의 지연을 의미하므로 성능에 미치는 영향을 최소화하기 위해 최소한으로 사양해야 한다.

## Using CUDA GPU Timers

CUDA 이벤트 API는 이벤트를 생성 및 삭제하고, 이벤트를 레코딩(timestamp 포함)하고 각 이벤트에서 timestamp의 차이를 msec 단위의 부동소수점 값으로 변환하는 함수를 제공한다. 

[Recording Events and Measuring Elapsed Time](/cuda/study/15_introducing_cuda_event.md#recording-events-and-measuring-elapsed-time)에서 API 사용 방식에 대해 자세히 다룬다.

CUDA 이벤트를 사용하여 시간을 측정하는 코드를 간단히 작성하면 다음과 같다.
```c++
cudaEvent_t start, stop;
float time;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord( start, 0 );
kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y,
                           NUM_REPS);
cudaEventRecord( stop, 0 );
cudaEventSynchronize( stop );

cudaEventElapsedTime( &time, start, stop );
cudaEventDestroy( start );
cudaEventDestroy( stop );
```

`cudaEventRecord()`는 `start`와 `stop` 이벤트를 default 스트림에 레코딩한다. Device는 해당 스트림에서 이벤트에 도달했을 때, 해당 이벤트에 대한 타임스탬프를 기록한다. `cudaEventElapsedTime()` 함수는 `start`와 `stop` 이벤트의 기록 간에 경과된 시간을 반환한다. 이 값의 단위는 밀리초이며, 약 0.5us의 resolution을 갖는다. 시간 측정은 GPU 클럭에서 측정되므로 resolution은 OS와 무관하다.

<br>

# Bandwidth

Bandwidth는 데이터를 전송할 수 있는 속도를 의미하며, 성능에서 가장 중요한 평가 요소 중 하나이다. 최적화할 때, bandwidth에 어떻게 영향을 미치는지 고려하여 최적화해야 한다. [Memory Opimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)의 가이드에서 언급하는 것과 같이, 어던 메모리에 데이터를 저장하는지, 데이터 레이아웃을 어떻게 설정하는지, 어떤 순서로 액세스하는지 등등 여러 요소에 따라 bandwidth가 크게 달라진다.

## Theoretical Bandwidth Calculation

Theoretical bandwidth는 product 설명서에 제공되는 하드웨어 사양을 사용하여 계산할 수 있다. 예를 들어, NVIDIA Tesla V100은 메모리 클럭 속도가 877MHz이고 4096-bit-wide memory interface인 HBM2 RAM(double data rate)을 사용한다.

이 데이터를 사용하면, NVIDIA Tesla V100의 peek theoretical bandwidth는 898 GB/s로 계산된다.

$$ (0.877 \times 10^9 \times (4096/8) \times 2) \div 10^9 = 898\text{ GB/s} $$

위 식에서 메모리 클럭은 Hz 단위이며, memory interface는 바이트 단위이고 마지막 `x2`는 double data rate에 의해서 곱해진 것이다. 마지막으로 $10^9$로 나누어 결과를 `GB/s` 단위로 변환한다.

## Effective Bandwidth Calculation

Effective bandwidth는 특정 작업이 걸린 시간과 이 작업에서 얼마나 데이터가 액세스되었는지 알고 있으면 계산할 수 있다. 이를 계산하기 위해서 다음의 공식을 사용할 수 있다.

$$ \text{Effective bandwidth } = ((B_r + B_w) \div 10^9) \div \text{ time}  $$

여기서 effective bandwidth의 단위는 `GB/s`이며, $B_r$은 커널 당 bytes read의 수이고 $B_w$는 커널 당 bytes write의 수이다. 경과 시간의 단위는 초(seconds)이다.

예를 들어, 2048 x 2048 행렬 복사의 effective bandwidth는 다음과 같이 계산할 수 있다.

$$ \text{Effective bandwidth } = ((2048^2 \times 4 \times 2) \div 10^9) \div \text{ time} $$

요소의 갯수($2048^2$)에 각 요소의 크기(4 bytes for a float)와 2(read/write)가 곱해졌다.

## Throughput Reported by Visual Profiler

Compute capability 2.0 이상의 device에서 visual profiler, 최근 device에서는 `nsight compute`를 통해 다양한 memroy throughput 측정을 수집할 수 있다. 여기서 따로 자세히 다루지는 않는다.

> Minimum memory transaction 크기는 대부분의 word 크기보다 크기 때문에, 실제 memory throughput은 커널에서 사용되지 않는 data transfer까지 포함한다.

<br>

# References

- [NVIDIA CUDA Documentation: Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)