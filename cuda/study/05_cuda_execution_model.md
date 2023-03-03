# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [GPU Architecture Overview](#gpu-architecture-overview)
  - [Warp Schedulers and Instruction Dispatch Units](#warp-schedulers-and-instruction-dispatch-units)
  - [Concurrent Kernel Execution](#concurrent-kernel-execution)
  - [Dynamic Parallelism](#dynamic-parallelism)
  - [Hyper-Q](#hyper-q)
- [References](#references)

<br>

# Intro

CUDA programming model에는 두 가지 중요한 추상화(**memory hierarhcy** and **thread hierarhcy**)가 있는데, 이를 통해 거대한 병렬 GPU를 제어할 수 있다. **CUDA execution model**에서는 GPU 병렬 구조를 추상적인 관점으로 볼 수 있다. 이 구조를 살펴보면 어떻게 스레드들이 병렬로 동작하는지 이해할 수 있다. 그리고, instruction throughput과 memory accesses 관점에서 어떻게 효율적인 코드를 작성할 수 있는지에 대한 인사이트를 얻을 수 있다.

<br>

# GPU Architecture Overview

GPU 아키텍처는 확장 가능한(scalable) **Streaming Multiprocessors (SMs)** 로 구성되며, GPU harware parallelism은 SM 빌딩 블록의 복제(replication)을 통해 달성된다. 아래 리스트는 GPU 아키텍처의 주요 컴퍼넌트들이다.

- CUDA Cores
- Shared Memory / L1 Cache
- Register File
- Load/Store Units
- Special Function Units
- Data Processing Units
- Warp Scheduler

전형적인 NVIDIA GPU는 아래와 같은 구조를 가지고 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcdgz8q%2FbtrYUTOD2Xg%2FcAVaGpSiJHdNmTykbRAQNk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

GPU 내 각 SM은 수백 개의 스레드의 동시 실행을 서포트하도록 설계되어 있고, 일반적으로 GPU에는 여러 개의 SM이 있으므로 하나의 GPU에서 수천 개의 스레드가 동시에 실행될 수 있다. 하나의 커널 그리드(kernel grid)가 실행(launch)되면, 이 그리드의 스레드 블록들은 이용할 수 있는 SM에 분배된다. SM에 이 스레드 블록들이 스케쥴링되면, 스레드 블록의 스레드들이 할당된 SM에서 동시에 실행된다. 이때, 하나가 아닌 여러 스레드 블록을 같은 SM에 할당할 수도 있는데, 사용 가능한 SM 리소스에 따라서 스케쥴링된다.

> 우리는 커널을 실행할 때, 일반적으로 여러 스레드 블록으로 실행되도록 execution configuration을 지정한다. 이때, 전체적인 그림으로 봤을 때 실행된 커널의 모든 스레드들이 동시에 실행되는 것처럼 보인다. 하지만, 위 내용처럼 하드웨어 수준으로 살펴보면, 실제로 모든 스레드들이 동시에 병렬 실행되는 것이 아닌 스레드 블록 단위로 병렬 실행된다는 것을 알 수 있다.

CUDA는 SIMT(Single Instruction, Multiple Thread) 아키텍처를 활용한다. 이 아키텍처를 활용하여 CUDA는 32개의 스레드를 그룹화하여 관리하고 실행한다. 32개의 스레드 그룹을 **warp** 라고 부른다 (워프라고 발음하는 것 같음). 워프 내의 모든 스레드들은 동시에 동일한 명령을 실행한다. 각 스레드들은 각자 자신의 instruction address counter와 register state를 가지며, 현재 명령을 수행한다. SM들은 할당된 스레드 블록을 32개의 스레드로 구성된 워프로 분할하여 사용 가능한 하드웨어 리소스에 대해 실행하도록 스케쥴링하게 된다.

> SIMT 아키텍처는 SIMD(Single Instruction, Multiple Data) 아키텍처와 유사하다. SIMD와 SIMT는 모두 동일한 명령을 여러 execution units에 브로드캐스팅하여 병렬 처리되도록 한다. 중요한 차이점 중 하나는 SIMD는 벡터의 모든 요소가 동기화된 그룹으로 함께 실행되는 반면, SIMT는 동일한 워프 내의 여러 스레드들이 독립적으로 수행될 수 있다는 것이다. 따라서, 한 워프 내의 모든 스레드들이 동시에 실행되더라도, 각 스레드들은 서로 다른 동작을 수행할 수 있다. SIMD와 비교했을 때, SMIT 모델에서는 추가적으로 아래의 세 가지 기능을 제공한다.
> - Each thread has its own instruction address counter
> - Each thread has its own register state
> - Each thread can have an independent execution path

> **A Magic Number: 32**
> 
> CUDA 프로그래밍에서 32라는 수는 magic numbe이다. 이는 하드웨어에 의해 결정되는데, 소프트웨어의 성능에 아주 중요한 영향을 미친다.
> 
> 개념적으로는 SM이 SIMD 방식으로 동시에 처리하는 작업을 세분화했다고 생각하면 된다. 즉, 32개의 스레드들을 하나의 그룹(warp)으로 만들고, 이 그룹 단위로 처리한다고 보면 된다. 따라서, warp의 크기(32개의 스레드)에 딱 맞추도록 최적화하면 일반적으로 GPU 리소스를 보다 효율적으로 활용할 수 있다. 이에 대해서는 다른 포스팅에서도 다룰 예정.

한 스레드 블록은 오직 하나의 SM에 의해 스케쥴링된다. SM에 의해서 스레드 블록이 스케쥴링되면, 실행이 완료될 때까지 SM에 남아있는다. SM은 하나 이상의 스레드 블록을 동시에 홀드할 수 있다. 아래 그림은 CUDA 프로그래밍의 논리적 관점과 하드웨어 관점을 바라봤을 때, 어떻게 대응되는지 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcdJHOA%2FbtrYYpGgOJw%2FyNhgCESFHFvPkuX4jGI7t0%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

<br>

Shared memory와 registers는 SM에서 중요한 리소스이다. Shared memory는 SM에 상주하는 스레드 블록들 간에 분배되는 리소스이며, register는 스레드들 간에 분배되는 리소스이다. 같은 스레드 블록 내의 스레드들은 이 리소스들을 이용하여 서로 협력하거나 통신할 수 있다. 위에서 언급했듯이, 한 스레드 블록 내의 모든 스레드들은 논리적으로 병렬로 실행되지만, 물리적으로 모든 스레드들이 동시에 실행될 수 있는 것은 아니다. 즉, 한 스레드 블록 내의 어떤 다른 스레드들은 서로 다른 페이스로 진행될 수 있다는 것을 의미한다.

당연히 병렬 스레드 간 데이터를 공유하는 것은 race condition을 일으킬 수 있다. 여러 스레드가 동일한 데이터에 접근할 때 그 순서는 고정되어 있지 않으므로 예상할 수 없는 동작이 수행된다. CUDA는 같은 스레드 블록 내의 스레드들 간의 동기화 수단을 제공하여, 스레드 블록 내 스레드들이 다음 단계를 수행하기 전에 특정 위치에 모두 도달할 수 있도록 할 수 있다. 그러나 스레드 블록 간의 동기화(inter-block synchronization)는 제공되지 않는다. Compute Capability 9.0부터는 클러스터 내의 스레드 블록 간의 데이터 공유나 동기화를 지원하는 것 같다.

<br>

스레드 블록 내의 warps는 어떤 순서로도 스케쥴링될 수 있지만, **active warps** (실제 실행되는 warp)의 수는 SM 리소스에 의해 제한된다. 만약 어떤 이유로 (예를 들어, device memory로부터 read하는 것을 대기) warp가 idle이 되면, SM은 상주하고 있는 다른 스레드 블록으로부터 이용 가능한 warp를 스케쥴링한다. 하드웨어 리소스는 SM의 모든 스레드와 블록 간 파티셔닝되어 있으므로 warp를 스위칭하는 데에는 오버헤드가 없다.

<br>

## Warp Schedulers and Instruction Dispatch Units

각 SM에는 **warp shedulers** 와 **instruction dispatch units** 가 있다. GPU 아키텍처 세대마다 갯수는 다를 수 있는데, 여기서는 각 SM에 2개의 warp scheduler와 2개의 instruction dispatch units이 있다고 가정한다.

한 스레드 블록이 SM에 할당될 때, 스레드 블록의 모든 스레드들은 warp 단위로 나누어진다. 만약 스레드 블록 내에 128개의 스레드가 있다면 총 4개의 warp로 나누어진다. 이때, 두 개의 warp scheduler는 두 개의 warp를 선택하고 하나의 instruction을 issue한다. 이를 시각적으로 표현하면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcpssCN%2FbtrYT86mc4B%2F7fuxIcZHWx9MgRXrp8lCkK%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

<br>

## Concurrent Kernel Execution

Fermi 아키텍처부터 concurrent kernel execution을 지원한다. 이는 동일한 GPU에 대해 실행 중인 어플리케이션에서 여러 커널을 동시에 실행할 수 있다는 것을 의미한다. Concurrent kernel execution을 통해 프로그램은 수 많은 작은 커널들을 실행시켜 GPU를 최대로 활용할 수 있다. Fermi 아키텍처에서는 하나의 GPU에서 최대 16개의 커널을 동시에 실행할 수 있다. Concurrent kernel execution은 **CUDA Stream**과 연관되어 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFPOJX%2FbtrYTNOWKRI%2FbATq4W9Du5CysLrmEkCK3K%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

<br>

## Dynamic Parallelism

**Dynamic Parallelism** 은 Kepler 아키텍처에 도입된 새로운 기능이다. 이는 동적으로 새로운 그리드를 실행할 수 있도록 해준다. 즉, 어떤 커널 함수 내부에서 새로운 커널을 실행시킬 수 있다는 것을 의미한다. 시각적으로 표현하면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcbG3kO%2FbtrYTpm8od6%2FwbFJRsomjVqoZk6KkOkyaK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

<br>

## Hyper-Q

Kepler 아키텍처부터 **Hyper-Q** 를 지원한다. Hyper-Q는 CPU와 GPU 간에 더 많은 simultaneous hardware connection을 추가하여 CPU 코어가 GPU에서 더 많은 작업을 동시에 실행할 수 있도록 해준다. 결과적으로 GPU utilization을 증가와 CPU idle time 감소를 기대할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbezbFF%2FbtrYRRY6zHJ%2FPlafWzR5McqwMYaniQdLok%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

위 그림에서 보듯, Fermi 아키텍처는 CPU에서 GPU로 태스크를 전달할 때 하나의 hardware work queue를 사용한다. 따라서 하나의 태스크가 그 뒤에 있는 다른 모든 태스크가 진행되지 못하도록 블락하게 된다.

반면 hyper-Q는 이러한 제약이 없다. Kepler GPU는 32개의 hardware work queues를 제공한다. 이로 인해 GPU에서 더 많은 동시성(concurrency)을 제공하여 GPU utilization을 극대화하고 전반적인 성능을 향상시킬 수 있다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- (Ampere Architecture) [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- NVIDIA GPU 아키텍처 변천사 ([상편](http://www.donghyun53.net/nvidia-gpu-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-%EB%B3%80%EC%B2%9C%EC%82%AC-%EC%83%81%ED%8E%B8/), [하편](http://www.donghyun53.net/nvidia-gpu-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-%EB%B3%80%EC%B2%9C%EC%82%AC-%ED%95%98%ED%8E%B8/))
- https://pc.watch.impress.co.jp/docs/column/kaigai/755994.html
- https://www.nextplatform.com/2018/02/28/engine-hpc-machine-learning/