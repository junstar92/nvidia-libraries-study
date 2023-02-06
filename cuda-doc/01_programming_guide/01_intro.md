# Table of Contents
- [Table of Contents](#table-of-contents)
- [The Benefits of Using GPUs](#the-benefits-of-using-gpus)
- [CUDA : A General-Purpose Parallel Computing Platform and Programming Model](#cuda--a-general-purpose-parallel-computing-platform-and-programming-model)
- [A Scalable Programming Model](#a-scalable-programming-model)
- [References](#references)

<br>

# The Benefits of Using GPUs

GPU (The Graphics Processing Unit)은 비슷한 가격과 파워의 CPU 대비 더 높은 instruction throughput과 memory bandwidth를 제공한다. 그래서 많은 연산이 필요한 프로그램에서는 CPU보다 GPU를 활용하여 더 빠르게 실행시킬 수 있다. FPGAs와 같은 디바이스 또한 에너지 효율이 좋지만, GPU보다 프로그래밍 유연성이 좋지는 않다.

CPU와 GPU의 차이는 애초에 설계 목적이 다르기 때문이다. CPU는 스레드(thread)라는 일련의 명령을 가능한 빠르게 실행하고 10개 가량의 스레드를 병렬로 실행하도록 설계된 반면, GPU는 수 천개의 스레드를 병렬로 실행하는데 특화되도록 설계되어 있다. GPU의 단일 스레드의 성능은 CPU보다 많이 느리지만, 물량으로 밀어붙여 더 큰 처리량을 달성한다고 볼 수 있다.

GPU는 극도의 병렬 연산에 특화되어 있어서 수 많은 트랜지스터들이 데이터 캐싱이나 제어 흐름보다 데이터 처리에 사용하도록 설계되어 있다. 아래 그림은 CPU와 GPU의 칩이 어떻게 구성되어 있는지를 대략적으로 보여준다.

왼쪽: CPU(multicore) / 오른쪽: GPU(manycore)

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png" height=300px width=auto style="display: block; margin: 0 auto"/>

앞서 언급했듯이 CPU는 순차 코드(sequential code)의 성능을 최적화하도록 설계되어 있고, 명령과 데이터 액세스의 지연 시간(latency)를 줄이기 위해 GPU보다 상대적으로 큰 캐시 메모리를 많이 사용한다. 반면 GPU는 부동소수점 연산과 같은 데이터 처리에 특화된 많은 트랜지스터들을 가지고 있기 때문에 고도의 병렬 연산에 유리하다. 대신 메모리에 액세스하는 지연 시간이 CPU보다 길지만, 연산을 통해 이를 드러나지 않도록 한다.

일반적인 프로그램에서는 병렬 처리와 순차 처리가 혼재되어 있으므로 전체적인 성능을 끌어올리기 위해 CPU와 GPU를 함께 사용하도록 시스템을 설계한다.

<br>

# CUDA : A General-Purpose Parallel Computing Platform and Programming Model

2006년 11월, NVIDIA에서 CUDA를 발표했다. 이를 활용하면 NVIDIA GPU를 통해 CPU보다 효율적으로 복잡한 계산 문제들을 해결할 수 있다. CUDA는 C++을 사용하여 개발할 수 있는 환경을 갖추고 있고, 아래 그림에서 보여주는 것처럼 다른 언어, API, 또는 지시어 기반 방식도 지원한다.

<img src="https://docs.nvidia.com/cuda/archive/11.2.0/cuda-c-programming-guide/graphics/gpu-computing-applications.png" height=500px width=auto style="display: block; margin: 0 auto"/>

<br> 


# A Scalable Programming Model

CUDA 프로그래밍 모델은 C에 익숙한 개발자들이 쉽게 익힐 수 있도록 설계되었다. CUDA에는 세 가지 핵심이 있는데, 이는 다음과 같다.

- **스레드 계층** (a hierarchy of thread groups)
- **공유 메모리** (shared memories)
- **배리어 동기화** (barrier synchronization)

이 세 가지 핵심은 C/C++ 언어에서 최소한의 확장을 통해 개발자에게 제공된다.

CUDA 프로그래밍에 대해 본격적으로 살펴보면 쉽게 이해할 수 있는 부분인데, 간단히 설명하면 세 가지 핵심을 통해 데이터 병렬화(data parallelism)와 스레드 병렬화(thread parallelism)을 제공한다. 우리는 주어진 문제를 하위 문제로 분할하고, 각 하위 문제들을 더 세부적으로 나누어서 스레드들이 나뉘어진 문제들을 병렬적으로 해결하도록 해야 한다.

문서에서 언급하는 **확장 가능한 프로그래밍 모델(scalable programming model)** 이라는 용어를 간단히 요약하면 어떤 NVIDIA GPU를 사용하더라도 코드를 동일하게 사용할 수 있다는 것을 의미한다. 병렬 연산을 실행할 때, 실제로 CUDA는 스레드 블록들이 알아서 수행되도록 스케쥴링하며, GPU 내의 multiprocessor 갯수에 상관없이 CUDA 프로그램을 실행할 수 있다 (엄밀히 말하자면 각 GPU마다 조건은 존재한다). 이를 그림으로 표현하면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeqSDTH%2FbtrmDQvtMn3%2F4wceGUzPsWg6oaV5gjN1ZK%2Fimg.png" height=400px width=auto style="display: block; margin: 0 auto"/>

여기서 컴파일된 CUDA 프로그램 내에는 8개의 스레드 블록을 사용하도록 구성되어 있는데, 서로 다른 수의 SM(streaming multiprocessor)를 갖는 GPU에서 코드 수정없이 실행될 수 있다는 것을 보여준다. 직관적으로 다가오지 않더라도, CUDA 프로그래밍을 알아가다보면 이 부분은 자연스럽게 이해되는 부분이다.

> [Heterogeneous Computing](/cuda-study/01_heterogeneous_computing.md)에서 이와 관련된 내용에 대해 추가로 다루고 있음

<br>

# References
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#)