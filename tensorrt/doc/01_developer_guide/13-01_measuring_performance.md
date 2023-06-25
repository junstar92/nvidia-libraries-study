# Table of Contents

- [Table of Contents](#table-of-contents)
- [Mearsuring Performance](#mearsuring-performance)
- [Wall-clock Timing](#wall-clock-timing)
- [CUDA Events](#cuda-events)
- [Built-In TensorRT Profiling](#built-in-tensorrt-profiling)
- [CUDA Profiling Tools](#cuda-profiling-tools)
  - [Profiling for DLA](#profiling-for-dla)
- [Tracing Memory](#tracing-memory)
- [References](#references)

<br>

# Mearsuring Performance

모든 최적화가 그러하듯 TensorRT로 최적화 작업을 시작하기 전에 무엇을 측정해야 하는지 결정하는 것이 중요하다.

### Latency

네트워크 추론에 대한 성능 지표 중 하나는 입력이 네트워크로 전달되고 출력을 이용 가능할 때까지 경과되는 시간(time elapes)이다. 이는 단일 추론(single inference)에 대한 네트워크의 `latency`이며, 짧으면 짧을수록 좋다.

### Throughput

또 다른 성능 지표는 고정된 단위 시간동안 얼마나 많은 추론을 완료할 수 있는지이다. 이는 네트워크의 `throughput`이며, 높으면 높을수록 좋다. Throughput이 높을수록 컴퓨터 리소스를 효율적으로 활용한다는 것을 나타낸다.

<br>

Latency와 throughput을 확인하는 또 다른 방법은 최대 latency를 고정하고 해당 시간에서의 throughput을 처리하는 것이다. 이와 같은 성능 측정은 사용자 경험과 시스템 효율성 간의 합리적인 절충안이 될 수 있다.

Latency와 throughput을 측정하기 전에 측정의 시작과 끝을 정확하게 선택해야 한다.

많은 어플리케이션에는 파이프라인이 있고, 전체 시스템의 성능은 전체 파이프라인의 latency와 throughput으로 측정될 수 있다. 전처리 및 후처리 단계는 어플리케이션 구현에 크게 의존하므로 이 포스팅에서는 네트워크 추론의 latency와 throughput만 고려한다.

<br>

# Wall-clock Timing

`Wall-clock time`은 연산의 시작과 끝 지점 간의 경과 시간이며, 어플리케이션의 전체적인 throughput과 latency를 측정하는데 유용할 수 있다. C++11에서는 `<chrono>` 표준 라이브러리에서 high precision timers를 제공한다. 예를 들어, `std::chrono::system_clock`은 시스템의 wall-clock time을 나타내고, `std::chrono::high_resolution_clock`은 사용 가능한 가장 높은 정밀도로 시간을 측정한다.

아래 예제 코드는 네트워크 추론의 host time을 측정하는 방법을 보여준다.
```c++
#include <chorno>

auto startTime = std::chrono::high_resolution_clock::now();
context->enqueueV3(stream);
cudaStreamSynchronize(stream);
auto endTime = std::chrono::high_resolution_clock::now();
float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
```

위의 방법은 한 번에 하나의 추론만 발생하는 경우에 사용할 수 있는 간단한 프로파일링 방법이다. 추론은 일반적으로 비동기식이므로 명시적으로 CUDA 스트림이나 device 동기화로 결과를 사용할 수 있을 때까지 기다려야 한다.

<br>

# CUDA Events

Host에서의 시간 측정은 host/device synchronization이 필요하다는 문제를 갖는다. 최적화된 어플리케이션은 데이터가 이동될 때 병렬로 연산이 실행되는 추론이 있을 수 있다. 또한, 동기화 연산 자체는 타이밍 측정에 약간의 노이즈를 추가하게 된다.

이러한 문제를 해결하기 위해 CUDA는 [Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__EVENT)를 제공한다. 이 API를 사용하면 이벤트가 발생할 때 GPU에 의해 타임스탬프가 기록되는 CUDA 스트림에 이벤트를 배치할 수 있다. 그러면 타임스탬프의 차이를 통해 서로 다른 연산이 얼마나 오래 걸렸는지 알 수 있다.

아래 예제 코드는 두 CUDA 이벤트 사이의 시간 계산을 보여준다.
```c++
cudaEvent_t start, endl;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
context0->enqueueV3(stream);
cudaEventRecord(end, stream);

cudaEventSynchronize(end);
float totalTime;
cudaEventElapsedTime(&totalTime, start, end);
```

<br>

# Built-In TensorRT Profiling

추론 성능을 조금 더 깊게 살펴보려면 최적화된 네트워크 내에서 더 세분화된 성능 측정이 필요하다.

TensorRT에는 TensorRT가 프로파일링 정보를 어플리케이션으로 전달하도록 구현할 수 있는 `Profiler` 인터페이스가 있다. 이 인터페이스가 호출되면 네트워크는 프로파일링 모드에서 실행된다. 추론이 끝나면 프로파일러 객체가 호출되어 각 레이어에서의 측정된 시간을 리포트한다. 이를 사용하여 병목을 찾고 다른 버전의 직렬화된 엔진들을 비교하여 성능 문제를 디버깅할 수 있다.

프로파일링 정보는 일반적인 `enqueueV3()` 실행의 추론 또는 CUDA graph 실행에서 수집할 수 있다. 이와 관련된 정보는 `IExecutionContext::setProfiler()`와 `IExecutionContext::reportToProfiler()`에서 살펴볼 수 있다.

루프(loop) 내부의 레이어는 single monolithic layer로 컴파일되므로 해당 레이어에 대한 별도의 시간 측정은 사용할 수 없다. 또한 일부 서브그래프(특히 Transformer와 같은 네트워크)는 아직 Profiler APIs에 통합되지 않은 next-generation graph optimizer에서 처리된다. 이러한 네트워크의 경우에는 CUDA 프로파일링 도구를 사용하여 프로파일링할 수 있다.

`IProfiler` 인터페이스를 사용하는 방법은 샘플 코드의 [common.h](https://github.com/NVIDIA/TensorRT/blob/main/samples/common/common.h#L123)를 통해 확인할 수 있다.

TensorRT와 함께 제공되는 `trtexec`를 사용하여 주어진 네트워크 및 플랜 파일에서의 성능을 측정할 수도 있다.

<br>

# CUDA Profiling Tools

CUDA 프로파일러로 [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)이 권장된다. 이전에 사용하던 `nvprof`와 `nvvp`는 deprecated 되었다. Nsight Systems를 사용하면 모든 CUDA 프로그램에서 실행 중에 시작된 커널에 대한 timing 정보, host와 device 간의 데이터 이동 및 사용된 CUDA API 호출 정보를 살펴볼 수 있다. 또한, GPU 정보와 함께 기존의 CPU 샘플링 프로파일 정보도 리포트는 등 다양한 방식으로 구성할 수 있다.

### Profile Only the Inference Phase

TensorRT 어플리케이션을 프로파일링할 때, 엔진이 빌드된 이후에만 프로파일링을 활성화해야 한다. Build phase에서는 모든 가능한 tactics가 시도되고 시간이 측정된다. 이 부분에 대한 프로파일링은 의미가 없으며 추론을 위해 실제로 선택된 커널이 아닌 모든 커널이 성능 측정에 포함된다. 프로파일링 범위를 제한하는 한 가지 방법은 다음과 같다.

- **First phase:** 엔진 빌드 및 직렬화(serialization).
- **Second phase:** 직렬화된 엔진을 로드하고 추론을 실행. 이 단계에서만 프로파일링을 수행.

만약 어플리케이션이 위의 두 단계를 연속적으로 실행해야 하는 경우, 두 번째 단계 양 끝에 `cudaProfilerStart()`/`cudaProfilerStop()`의 CUDA API를 추가하고 Nsight Systems 커맨드에 `-c cudaProfilerApi` 플래그를 추가하여 프로파일링할 수 있다. 그러면 해당 API 호출 사이의 부분만 프로파일링된다.

### Understand Nsight Systems Timeline View

Nsight System의 타임라인에서 GPU activities는 `CUDA HW` 아래의 행에서 보여주고 CPU activities는 `Threads` 아래의 행에서 보여준다.

전형적인 추론 과정에서 어플리케이션은 `context->enqueueV3()` 또는 `context->executeV2()` API를 호출하여 jobs를 큐에 삽입하고 GPU가 이 jobs가 완료할 때까지 기다리기 위해 스트림에 동기화한다. CPU activities를 살펴보면 `cudaStreamSynchronize()` 호출에서 시스템이 아무것도 하지 않는 것처럼 보일 수 있지만, 실제로 GPU는 CPU가 기다리는 동안 큐에 추가된 작업을 실행하느라 바쁠 수 있다. 아래 그림은 추론 시의 타임라인 예시를 보여준다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/timeline-view.png" height=700px style="display: block; margin: 0 auto; background-color:white"/>

### Use the NVTX Tracing in Nsight Systems

Nsight Compute와 Nsight Systems에 NVTX Tracing을 활성화하여 TensorRT 어플리케이션에서 생성된 데이터를 수집할 수 있다. NVTX는 어플리케이션에서 events와 ranges를 표시하기 위한 C-based API이다.

TensorRT는 NVTX를 사용하여 각 레이어의 범위를 표시하고 CUDA 프로파일러를 통해 각 레이어 범위에서 호출된 커널이 무엇인지 볼 수 있다. NVTX를 통해 런타임 엔진에서의 레이어 실행과 호출된 CUDA 커널을 서로 연관시킬 수 있다. Nsight Systems에서는 타임라인에서 이러한 event와 range 데이터를 수집하고 시각화한다.

TensorRT에서 각 레이어는 작업을 수행하기 위해 하나 이상의 커널을 실행할 수 있다. 실행되는 커널은 최적화된 네트워크와 하드웨어에 따라 다르다. Builder의 선택에 따라서 레이어 연산과 함께 데이터를 재 정렬하는 여러 추가 작업이 있을 수 있다.

아래 그림은 CPU 측에서 실행된 레이어의 실행 및 커널을 보여준다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/layer-exe.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

실케 커널은 GPU에서 실행된다. 아래 이미지는 CPU 측에서의 레이어와 커널 실행과 GPU 측에서의 실행 사이의 상관 관계를 보여준다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/kernels-gpu.png" height=700px style="display: block; margin: 0 auto; background-color:white"/>

<br>

### Control the Level of Details in NVTX Tracing

기본적으로 TensorRT는 NVTX markers에서 레이어 이름만 보여준다. 이는 사용자가 엔진을 빌드할 때 `IBuilderConfig`에서 `ProfilingVerbosity`를 설정하여 제어할 수 있다. 예를 들어, NVTX Tracing을 비활성화하려면 아래와 같이 `ProfilingVerbosity`를 `kNONE`으로 설정한다.
```c++
config->setProfilingVerbosity(ProfilingVerbosity::kNONE);
```

반면, 세부적인 레이어 정보(input and output dimensions, operations, parameters, tactic numbers, ...)를 출력하도록 하려면 `kDETAILED`로 설정하면 된다.
```c++
config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
```

<br>

### Run Nsight Systems with `trtexec`

아래 커맨드는 `trtexec`를 사용하여 Nsight Systems로 프로파일링하는 커맨드 예시이다.
```
trtexec --onnx=foo.onnx --profilingVerbosity=detailed --saveEngine=foo.plan
nsys profile -o foo_profile --capture-range cudaProfilerApi trtexec --profilingVerbosity=detailed --loadEngine=foo.plan --warmUp=0 --duration=0 --iterations=50
```

첫 번째 커맨드는 `foo.plan`으로 엔진을 빌드 및 직렬화하고, 두 번째 커맨드는 `foo.plan`을 사용하여 추론을 실행하고, Nsight Systems에서의 시각화를 위한 `foo_profile.nsys-rep` 파일을 생성한다.

`--profilingVerbosity=detailed` 플래그는 TensorRT가 NVTX marking에서 더 자세한 레이어 정보를 보여주도록 하며, `--warmUp=0 --duration=0 --interations=50` 플래그는 추론을 얼마나 반복 실행할 지를 제어할 수 있도록 한다. 기본적으로 `trtexec`는 3초 동안 추론을 실행하기 때문에 매우 큰 `nsys-rep` 파일이 생성될 수 있다.

만약 CUDA graph가 활성화되었따면, `--cuda-graph-trace=node` 플래그를 `nsys` 커맨드에 추가하여 per-kernel runtime 정보를 볼 수 있다.
```
nsys profile -o foo_profile --capture-range cudaProfilerApi --cuda-graph-trace=node trtexec --profilingVerbosity=detailed --loadEngine=foo.plan --warmUp=0 --duration=0 --iterations=50 --useCudaGraph
```

<br>

### (Optional) Enable GPU Metrics Sampling in Nsight Systems

Discrete GPU 시스템에서 `--gpu-metrics-device all` 플래그를 `nsys` 커맨드에 추가하면 GPU clock frequencies, DRAM bandwidth, Tensor Core utilization 등의 GPU 메트릭을 샘플링할 수 있다.

## Profiling for DLA

DLA를 프로파일링하려면 `--accelerator-trace nvmedia` 플래그를 `nsys` 커맨드에 추가하거나 GUI 환경에서는 `Collect other accelerators trace`를 활성화하면 된다. 예를 들어, CLI 환경에서는 다음과 같이 커맨드를 사용한다.
```
nsys profile -t cuda,nvtx,nvmedia,osrt --accelerator-trace=nvmedia  --show-output=true trtexec --loadEngine=alexnet_int8.plan --warmUp=0 --duration=0 --iterations=20
```

리포트 예시는 다음과 같다.

- `NvMediaDLASubmit`은 각 DLA subgraph에 대한 DLA task이다. DLA tasks의 런타임은 **Other accelerators trace** 아래의 DLA 타임라인에서 볼 수 있다.
- GPU fallback이 허용되기 때문에 몇몇 CUDA 커널이 TensorRT에 의해 자동으로 추가된다 (`permutationKernelPLC3`과 `copyPackedKernel`). 이들은 data reformatting에 사용되는 커널들이다.
- EGLStream APIs는 TensorRT가 GPU 메모리와 DLA 간의 데이터 전송을 위해 EGLStreams를 사용하기 때문에 실행된다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/profiling-dla-2.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

<br>

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/profiling-dla-1.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

<br>

# Tracing Memory

메모리 사용량 추적은 실행 성능만큼 중요할 수 있다. 일반적으로 메모리는 host보다 device에서 더 제한적이다. Device memory를 추적하기 위한 권장되는 메커니즘은 내부적으로 일부 통계(statistics)를 유지하고 `cudaMalloc` 및 `cudaFree`를 사용하는 간단한 커스텀 GPU allocator를 생성하는 것이다.

커스텀 GPU allocator는 `IGpuAllocator` API를 사용하여 네트워크 최적화에서 builder `IBuilder`와 엔진을 역직렬화할 때 `IRuntime`에 설정될 수 있다. 커스텀 allocator는 할당된 현재 메모리 양을 추적하고 allocation events의 전체 리스트에 타임스탬프 및 기타 정보가 있는 allocation event를 푸시할 수 있다. 이를 통해 시간 경과에 따른 메모리 사용량을 프로파일링할 수 있다.

모바일 플랫폼에서 GPU와 CPU 메모리는 시스템 메모리를 공유한다. 매우 제한된 메모리 크기의 디바이스에서는 비록 필요한 GPU 메모리가 시스템 메모리보다 작더라도 시스템 메모리가 부족할 수 있다. 이 경우에는 system swap size를 늘리면 일부 문제를 해결할 수 있다. 예제 스크립트는 다음과 같다.
```
echo "######alloc swap######"
if [ ! -e /swapfile ];then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo /bin/sh -c 'echo  "/swapfile \t none \t swap \t defaults \t 0 \t 0" >> /etc/fstab'
    sudo swapon -a
fi
```

<br>

# References

- [NVIDIA TensorRT Documentation: Measuring Performance](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#measure-performance)