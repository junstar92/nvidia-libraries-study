# Table of Contents

- [Table of Contents](#table-of-contents)
- [Kernels](#kernels)
- [Thread Hierarchy](#thread-hierarchy)
  - [Thread Block Clusters ](#thread-block-clusters-)
- [Memory Hierarchy](#memory-hierarchy)
- [Heterogeneous Programming](#heterogeneous-programming)
- [Asynchronous SIMT Programming Model ](#asynchronous-simt-programming-model-)
- [Compute Capability](#compute-capability)
- [References](#references)

이번 챕터에서는 CUDA 프로그래밍 모델의 컨셉을 살펴보고 C/C++에서 어떻게 사용할 수 있는지 알아본다.

<br>

# Kernels

CUDA에서는 C++을 확장하여 커널(kernel)이라고 불리는 C++ 함수를 정의할 수 있다. 이렇게 정의된 커널 함수가 호출되면 한 번 실행되는 일반 C++ 함수와 달리, N개의 다른 CUDA 스레드들이 병렬로 N번 함수를 실행한다.

커널 함수는 `__global__`라는 선언 지정자(declaration specifier)를 사용하여 정의되며, 이 커널 함수를 호출할 때는 CUDA의 `<<<...>>>`로 작성하는 execution configuration 문법을 사용하여 커널 함수을 수행할 CUDA 스레드의 수를 지정한다. 커널 함수를 실행하는 각 스레드에는 각자 고유한 ID가 있고, ID는 CUDA 내장 변수를 통해 커널 내에서 알아낼 수 있다.

벡터 덧셈을 예제로 살펴보자.

```c++
void vectorAdd(float const* a, float const* b, float* c, int const num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i];
    }
}
```
벡터 덧셈을 일반 순차 코드로 작성하면 위와 같이 구현할 수 있다. 하지만, CUDA kernel로 구현하면 CUDA 스레드 하나가 한 번의 덧셈을 처리하도록 구현하여 병렬화할 수 있다. 이때, 각 CUDA 스레드는 자신이 몇 번째 데이터를 처리해야 하는지 식별해야 하는데, 이를 위해 CUDA 내장 변수를 사용하게 된다.

간단히 `vectorAddKernel`를 다음과 같이 구현하고, `main` 내에서 [execution configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration) 문법을 사용하여 호출할 수 있다.
```c++
// kernel definition
__global__ void vectorAddKernel(float const* a, float const* b, float* c, int const num_elements)
{
    int i = threadIdx.x;

    if (i < num_elements) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    ...
    // kernel invocation with N threads
    vectorAddKernel<<<1, N>>>(a, b, c, num_elements);
    ...
}
```

위 코드는 N개의 스레드가 vectorAddKernel 함수를 실행하고, 각 스레드는 한 쌍의 덧셈을 수행한다.

> `vectorAddKernel` 예제에 대한 전체 코드는 [vector_add.cu](/code/cuda/vector_add/vector_add.cu) 를 참조 바람.

<br>

# Thread Hierarchy

CUDA 내장 변수인 `threadIdx`는 3차원 벡터이다. 따라서, CUDA 스레드는 1차원/2차원/3차원 스레드 인덱스를 사용하여 식별될 수 있으며, 각각 1차원/2차원/3차원 **스레드 블록(thread block)**을 형성할 수 있다. 여기서 스레드 블록은 일련의 스레드들의 집합이며 (스레드 )블록(block)이라고 부른다. 이를 이용하여 벡터, 행렬, 부피 연산에서 각 요소에 대해 자연스럽게 액세스할 수 있다.

스레드의 인덱스와 스레드 ID는 간단한 방식으로 서로 연관되어 있다. 1차원 블록의 경우에서는 인덱스와 ID는 서로 같다. 하지만 2차원 블록 (Dx, Dy)인 경우, 인덱스가 (x, y)인 스레드의 스레드 ID는 (x + y * Dx) 이다. 크기가 (Dx, Dy, Dz)인 3차원 블록의 경우, 인덱스가 (x, y, z)인 스레드의 ID는 (x + y * Dx + z * Dx * Dy) 이다.

NxN 크기의 두 행렬 A와 B의 덧셈의 결과를 행렬 C에 저장하는 경우, 간단하게 스레드 인덱스를 사용하여 다음과 같이 커널 함수를 구현할 수 있다.
```c++
// kernel definition
__global__
void matrixAddKernel(float a[N][N], float b[N][N], float c[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    c[i][j] = a[i][j] + b[i][j];
}

int main()
{
    ...
    // kernel invocation with one block of N * N * 1 threads
    int num_blocks = 1;
    dim3 threads_per_block(N, N);
    matrixAddKernel<<<num_blocks, threads_per_block>>>(a, b, c);
    ...
}
```

하드웨어와 연관된 이야기를 살짝 하면, 한 블록 내의 모든 스레드는 동일한 스트리밍 프로세서(streaming processor) 코어에 있어야 하고 코어의 제한된 메모리 리소스를 공유해야 한다. 따라서 한 블록당 가질 수 있는 스레드의 수는 제한된다. 현재 주로 사용되는 GPU에서 스레드 블록은 최대 1024개의 스레드를 가질 수 있다.

> 아직 살펴보지는 않았지만 하드웨어 계층과 함께 스레드 계층을 살펴보면, 하나의 스레드 블록은 하나의 스트리밍 프로세서 코어 내에서 수행된다. 그림으로 표현하면 아래와 같다.
> 
> <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2020/06/kernel-execution-on-gpu-1.png" height=300px style="display: block; margin: 0 auto"/>
> <br>


그러나, 커널은 동일한 모양의 여러 스레드 블록에서 실행될 수 있기 때문에 커널을 실행하는 스레드의 총 갯수는 블록 당 스레드 수에 블록의 수를 곱한 것과 같다. 블록은 아래 그림과 같이 1차원/2차원/3차원 그리드(grid)로 구성될 수 있다. 여기서 그리드는 스레드 블록들의 집합이며, 각 스레드 블록들은 모두 같은 크기를 가진다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png" height=200px style="display: block; margin: 0 auto"/>

그리드에서 스레드 블록의 수는 처리할 데이터의 크기에 따라 결정되며, 일반적으로 GPU의 프로세서 갯수를 넘어선다.

블록 당 스레드 수와 그리드랑 블록의 수는 `<<<...>>>` 문법을 통해 지정되며, `int` 또는 `dim3` 타입이어야 한다. 1차원인 경우에는 `int`를 사용하면 나머지 차원들은 모두 1이 기본값으로 지정되고, 2차원 이상인 경우 `dim3`를 사용하여 각 차원의 크기를 지정해주어야 한다. 바로 위의 코드에서 `matrixAddKernel` 커널을 호출할 때 사용하고 있다.

그리드 내의 각 블록 또한 1차원/2차원/3차원의 고유한 인덱스로 식별되고, 내장 변수인 `blockIdx`를 통해 알아낼 수 있다. 그리고 스레드 블록의 차원은 커널 내에서 `blockDim`이라는 내장 변수를 통해 알아낼 수 있다.

> 일반적으로 GPU에서 제공하는 스트리밍 프로세서 코어 수보다 많은 스레드 블록으로 커널을 호출하기 때문에, 사실 엄밀히 말하자면 처리할 데이터 전부가 병렬로 수행되는 것은 아니라는 것을 짐작할 수 있다.

이전에 구현한 `matrixAddKernel`에서 여러 스레드 블록을 처리할 수 있도록 구현하면 아래와 같이 구현할 수 있다.
```c++
// kernel definition
__global__
void matrixAddKernel(float a[N][N], float b[N][N], float c[N][N])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < N && j < N) {
        c[i][j] = a[i][j] + b[i][j];
    }
}

int main()
{
    ...
    // kernel invocation
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(N / threads_per_block.x, N / threads_per_block.y);
    matrixAddKernel<<<num_blocks, threads_per_block>>>(a, b, c);
    ...
}
```

위 코드에서 한 스레드 블록의 크기를 16x16 (256 threads)으로 설정하고 있다. 그리고 하나의 스레드가 행렬 요소 하나를 처리할 수 있도록 그리드의 차원을 설정하고 있다. 여기서는 단순히 표현하기 위해서 행렬 각 차원의 요소 갯수(N)가 블록의 각 차원의 스레드 갯수(16)으로 나누어 떨어진다고 가정하고 있으며, 만약 나누어 떨어지지 않는다면 해당 차원의 스레드 블록 수에 1을 더해주어야 모든 요소들을 커버할 수 있다.

하나의 스레드 블록은 하나의 스트리밍 프로세서 코어를 공유한다고 언급했었다. 따라서, 한 블록 내의 스레드들은 **공유 메모리(shared memory)**를 통해 데이터를 공유하여 협력할 수 있는데, 이때 메모리 액세스를 조정하기 위해 동기화(synchronization)가 필요하기도 한다. 공유 메모리를 잘 사용하게 되면 CUDA 커널의 성능을 더욱 향상시킬 수 있으므로 적절히 잘 사용하는 것이 필수적이며, 이에 대해서는 다른 챕터에서 다루도록 한다.

> 커널의 동기화 지점을 지정하려면 커널 내에서 내장 함수인 `__syncthreads()`를 호출하면 된다. `__syncthreads()`는 한 블록 내의 모든 스레드가 해당 위치에 도달할 때까지 기다리는 배리어 역할을 수행한다. 이외에도 [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) API를 통해 더욱 풍부한 스레드 동기화 방법을 제공한다.

## Thread Block Clusters <!-- TODO -->

> CUDA 11.8에서 추가된 내용

Compute Capability 9.0에서 도입된 기능이며, 필자도 아직 해당 GPU가 없어서 사용해보지는 못했다. 기존의 thread - block - grid의 계층에서 block과 grid 사이에 **thread block clusters** 라는 부가적인 계층 레벨을 도입하는 것이며, 클러스터라는 단위로 협력하기 위한 수단을 제공하는 것으로 보인다. 그림으로 표현하면 다음과 같다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png" height=200px style="display: block; margin: 0 auto"/>

> 자세한 내용은 추후에 GPU가 생기면 다루어 볼 예정

> 스레드 계층 구조에 대해 조금 더 자세히 살펴보고 싶다면 아래 링크를 참조하는 것도 도움이 될 수 있다.
>
> - [CUDA Thread 구조와 Data Mapping](https://junstar92.tistory.com/245)

<br>

# Memory Hierarchy

아래 그림과 같이 CUDA 스레드는 실행하는 동안 다양한 메모리 공간으로부터 데이터를 액세스할 수 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png" height=600px style="display: block; margin: 0 auto"/>

각 스레드는 private local memory를 가진다다. 각 스레드 블록은 블록 내 모든 스레드들이 접근할 수 있는 공유 메모리(shared memory)를 가지며, 한 블록 내 모든 스레드들에서 공유 메모리의 lifetime은 같다. 그리고 모든 스레드들은 같은 global memory에 액세스할 수 있다.

위 그림에서 thread block cluster에 대해서도 언급하고 있는데, 같은 클러스터 내의 스레드 블록은 서로의 공유 메모리에 read/write/atomic operations을 수행할 수 있다고 언급하고 있다. 이는 compute capability 9.0 GPU에 해당하는 내용이고, 현재 시점에서는 아마 거의 사용할 가능성은 낮을 것 같다.

그래서 일반적인 경우, 메모리 계층은 아래와 같다고 볼 수 있다.

<img src="https://docs.nvidia.com/cuda/archive/11.7.1/cuda-c-programming-guide/graphics/memory-hierarchy.png" height=500px style="display: block; margin: 0 auto"/>

<br>

# Heterogeneous Programming

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png" height=600px style="display: block; margin: 0 auto"/>

<br>

위 그림에서 설명하듯이, CUDA 프로그래밍 모델은 C++ 프로그램을 실행하는 **host** 의 coprocessor로서 분리된 디바이스에서 CUDA 스레드가 실행된다고 가정한다.

또한, CUDA 프로그래밍 모델에서 host(CPU)와 device(GPU)는 DRAM에서 각자 별도의 메모리 공간을 가진다고 가정하며, 이 메모리는 각각 host memory와 device memory라고 칭한다. 따라서, CUDA 런타임에서 device memory를 관리하며 이에 대해서는 다음 챕터에서 알아본다. CUDA 런타임에는 device memory 할당/해제, host와 device 간 메모리 전송이 포함된다.

참고로 host와 device memory를 연결하여 메모리를 관리하는 [**Unified Memory**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)를 제공하기도 하는데, 이를 사용하면 시스템 내에서 모든 CPU, GPU에서 액세스할 수 있는 메모리를 관리할 수 있다. 이를 통해 명시적으로 host와 device 간의 데이터 전송을 제거하여 코드를 단순화할 수 있는데, 이에 대한 내용도 다른 챕터에서 다룬다.

<br>

# Asynchronous SIMT Programming Model <!-- TODO -->

> CUDA 11.4에서 추가된 내용 (TODO)



<br>

# Compute Capability

**Compute capability** 는 version number를 나타내며, "SM version"으로 불리기도 한다. 이 버전은 GPU 하드웨어가 지원하는 기능을 식별하며, 런타임에서 특정 하드웨어 기능 또는 명령어가 현재 GPU에서 사용 가능한지 결정하는데 사용된다.

X.Y로 표기하며, X는 major version, Y는 minor version 이다.

동일한 major version의 GPU는 같은 core architecture 이다. Major version이 9인 경우 NVIDIA Hopper GPU 아키텍처 기반이고, 8은 NVIDIA Ampere GPU 아키텍처 기반, 7은 Volta 아키텍처 기반, 6은 Pascal 아키텍처, 5는 Maxwell 아키텍처, 3은 Kepler 아키텍처 기반이다.

Minor version은 새로운 기능을 포함하여 core architecture의 개선을 의미한다.

**Turing** 은 compute capability가 7.5인 장치의 아키텍처이며, Volta 아키텍처 기반에서 업데이트된 것이다.

각 compute capability의 technical specifications는 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)에서 확인할 수 있다.

<br>

# References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)