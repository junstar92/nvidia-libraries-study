# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introducing the CUDA Programming Model](#introducing-the-cuda-programming-model)
- [CUDA Programming Structure](#cuda-programming-structure)
- [Managing Memory](#managing-memory)
- [Organizing Threads](#organizing-threads)
- [CUDA Kernel](#cuda-kernel)
  - [How to launch a CUDA Kernel](#how-to-launch-a-cuda-kernel)
  - [How to write a CUDA Kernel](#how-to-write-a-cuda-kernel)
- [CUDA Error Handling](#cuda-error-handling)
- [Example: Vector Addition](#example-vector-addition)
- [References](#references)

<br>

# Introducing the CUDA Programming Model

> 기본적인 CUDA Programming Model에 대한 내용은 [Programming Model](/cuda-doc/01_programming_guide/02_programming_model.md)을 참조 바람

CUDA 프로그래밍 모델에서 GPU 아키텍처의 computing power를 활용하기 위해 가장 중요한 두 가지는 다음과 같다.

- GPU 계층 구조를 통해 스레드를 조직화하는 방법
- GPU 계층 구조를 통해 메모리에 액세스하는 방법

이 포스팅에서는 스레드를 조직화하는 방법에 대해서만 다룬다.

<br>

# CUDA Programming Structure

CUDA 프로그래밍 모델은 C언어에서 몇 가지 추가 기능을 제공하여 heterogeneous computing system(이종 컴퓨팅 시스템)에서 프로그램을 실행할 수 있도록 해준다. 이러한 시스템은 CPU와 CPU를 보완하는 GPU로 구성된다. CPU와 GPU는 각자 자신만의 메모리를 가지며, 메모리는 서로 PCI-Express bus로 연결된다. 이와 같은 관계에 의해서 이종 시스템 환경에서는 CPU와 GPU를 host와 device로 지칭한다.

- **Host** - the CPU and its memory(host memory)
- **device** - the GPU and its memory(device memory)

CUDA 프로그래밍 모델에서 중요한 컴포넌트는 커널(kernel)이다. 커널은 GPU에서 실행되는 코드이며, 순차 프로그램(sequential program) 코드처럼 커널 코드를 작성할 수 있다. Host 측에서는 단순히 알고리즘(kernel)을 data와 GPU device capability 기반으로 device에 매핑시켜주는 역할을 수행한다. 이를 통해, 수 천개의 스레드를 만들고 관리하는 세부 사항들에 얽매이지 않고 간단하게 순차 코드로 커널을 작성할 수 있도록 한다. 이종 환경 시스템 프로그램이 실행할 때, 그 이면에서 CUDA는 GPU 스레드에 대한 우리가 작성한 커널의 스케쥴링을 관리한다.

CUDA의 대부분 연산에서 host는 device와 독립적으로 동작될 수 있다. 예를 들어, Host 측에서 커널을 실행(공식 문서에 따르면 launch한다고 표현한다)하면, 커널은 호출된 즉시 제어권을 다시 host로 넘겨준다. 따라서, GPU device에서 data parallel code를 수행하는 동시에 CPU가 추가적인 태스크를 수행할 수 있도록 해준다. 즉, GPU는 대부분의 연산에서 CPU와 비동기로 동작한다는 것을 의미한다. 물론 비동기로 동작하지 않는 연산이나 조건들도 있다. 이러한 특징 때문에 GPU 연산은 host-device communication과 오버랩될 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcanbZp%2FbtrYeZDlf6X%2FwnAMNewats3DI8gnsK5r1k%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

전형적인 CUDA 프로그램은 serial code와 이를 보완하는 parallel code로 구성된다. 바로 위의 그림처럼 serial code(and task parallel code)는 host 측에서 실행되고, 반면 parallel code는 GPU device에서 실행된다.

CUDA 프로그램은 일반적으로 아래의 패턴으로 작성된다.

1. Copy data from CPU memory to GPU memory
2. Invoke kernel functions to operate on the data stored in GPU memory
3. Copy data back from GPU memory to CPU memory

<br>

# Managing Memory

공식 문서에서도 언급하지만, CUDA 프로그래밍 모델은 시스템이 host와 device로 구성되어 있다고 가정하고, host와 device는 각각 자신만의 분리된 메모리를 가지고 있다고 가정한다. CUDA runtime에서는 device memory를 할당/해제 또는 device <-> host 간의 데이터 전송 등을 위한 함수들을 제공하는데, 이를 활용하여 device를 제어하고 최적의 성능을 얻을 수 있다. 아래 표는 메모리 할당을 위한 C 함수와 대응되는 CUDA 함수를 보여준다.

|Standard C Functions|CUDA C Functions|
|--|--|
|`malloc`|`cudaMalloc`|
|`memcpy`|`cudaMemcpy`|
|`memset`|`cudaMemset`|
|`free`|`cudaFree`|

기본적으로 GPU 메모리 할당에 사용되는 함수는 `cudaMalloc`이며, 이 함수의 시그니처는 아래와 같다.

`cudaError_t cudaMalloc(void** devPtr, size_t size)`

이 함수는 **linear layout**으로 지정된 크기의 바이트만큼 할당한다.

Host <-> Device 간 데이터 전달 함수는 `cudaMemcpy` 이며, 이 함수의 시그니처는 아래와 같다.

`cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`

이 함수는 source memory 공간으로부터 지정된 바이트만큼 destination memory 공간으로 데이터를 복사한다. 데이터가 전달되는 방향은 `kind`로 지정하며, 지정할 수 있는 타입은 아래와 같다.

- `cudaMemcpyHostToHost` : Host -> Host
- `cudaMemcpyHostToDevice` : Host -> Device
- `cudaMemcpyDeviceToHost` : Device -> Host
- `cudaMemcpyDeviceToDevice` : Device -> Device
- `cudaMemcpyDefault` : Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing

사실 데이터 복사 방향이나 메모리 종류에 따라 조금씩 다르지만, 우선 일반적으로 `cudaMemcpy` 함수는 데이터 복사가 완료되면 return하는 **synchronous**으로 동작한다고 보면 된다. 동기화 동작과 관련한 내용은 공식 문서([link](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior))에서 자세히 알 수 있다.

`cudaMalloc`과 `cudaMemcpy`를 보면 알 수 있듯이, kernel launch를 제외한 모든 CUDA 함수는 `cudaError_t`라는 에러 코드를 리턴한다. 연산이 성공적이라면 `cudaSuccess`가 리턴되며, 이외에는 해당하는 에러 코드가 리턴된다. 참고로, CUDA 함수 중에 `udaGetErrorString`를 사용하면 human-readable error message를 얻을 수 있다. 이 함수의 시그니처는 다음과 같다.

`char* cudaGetErrorString(cudaError_t error)`

[vector_add.cu](/code/cuda/vector_add/vector_add.cu) 예제 코드를 살펴보면, `cudaMalloc`과 `cudaMemcpy`를 어떻게 사용하는지 잘 보여준다. 아래는 예제 코드에서 실제 덧셈을 수행하는 `vectorAdd` 함수 구현이다. 

```c++
void vectorAdd(float const* a, float const* b, float* c, int const num_elements)
{
    // allocate the device input vectors a, b, c
    float *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, sizeof(float) * num_elements));

    // copy the host input vector a and b in host memory
    // to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, a, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, b, sizeof(float) * num_elements, cudaMemcpyHostToDevice));

    // allocate CUDA events for estimating elapsed time
    cudaEvent_t start, stop;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // Launch the vectorAddKernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);

    CUDA_ERROR_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));

    // copy the device result vector in device memory
    // to the host result vector in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_ERROR_CHECK(cudaMemcpy(c, d_c, sizeof(float) * num_elements, cudaMemcpyDeviceToHost));

    // verify that the result vector is correct
    printf("Verifying vector addition...\n");
    for (int i = 0; i < num_elements; i++) {
        if (fabs(a[i] + b[i] - c[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // compute performance
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    double flops = static_cast<double>(num_elements);
    double giga_flops = (flops * 1.0e-9f) / (msec / 1000.f);
    printf("Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f ops, "
            " WorkgroundSize = %u threads/block\n",
            giga_flops, msec, flops, threads_per_block);
    
    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    CUDA_ERROR_CHECK(cudaFree(d_c));
    // free CUDA event
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
}
```

<br>

# Organizing Threads

커널 함수가 host 측으로부터 호출될 때, 커널 함수는 device에서 실행된다. Device에서는 많은 스레드들이 생성되고, 각 스레드는 커널 함수를 실행하게 된다. CUDA 프로그래밍에서 이러한 스레드를 어떻게 구성하느냐가 매우 중요하다. CUDA에는 스레드 계층 추상화(thread hierarchy abstraction)를 프로그래밍으로 표현하고, 이를 사용하여 스레드를 구성할 수 있다. 이 계층은 아래 그림과 같이 두 단계이며, 그리드(grid)를 블록(block)으로, 블록을 스레드로 분해할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpngIl%2FbtrYjDMVJDa%2FbHnlsqN1WLMFYtKutFncZk%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

> 위 그림에서 일반적으로 스레드 하나가 데이터 하나를 처리하고, GPU는 이 스레드들이 병렬로 수행된다고 볼 수 있다. (개념적으로 이렇다는 것이고, 이는 엄밀히 이야기하자면 정확한 내용은 아니다) 또한, CUDA 11.8에서 **thread block cluster**라는 계층이 추가되었다.

하나의 커널 실행(launch)로부터 생성된 모든 스레드들은 한 그리드에 속하게 된다. CUDA 메모리에 대해 살펴볼 때 더 자세히 보겠지만, 그리드 내의 모든 스레드들은 같은 global memory를 공유하게 된다.

그리드는 여러 블록들로 구성되는데, 하나의 스레드 블록은 스레드들의 집합이며 이들은 서로 block-level의 동기화(synchronization)와 shared memory를 통해 커뮤니케이션을 할 수 있다. 여기서는 일단 한 스레드 블록 내에 있는 스레드들이 서로 통신할 수 있는 방법이 있다는 것만 알고 있으면 된다. 서로 다른 스레드 블록에 있는 스레드들은 서로 통신할 수 없다 (cluster 내에 있는 스레드들은 가능).

스레드들은 고유한 좌표를 통해 서로를 구분할 수 있다.

- `blockIdx` : block index within a grid
- `threadIdx` : thread index within a block

CUDA 커널 내에서 액세스할 수 있는 위의 두 변수는 CUDA 내장 변수이다. 커널 함수가 실행되면, CUDA runtime에 의해서 각 스레드에 `blockIdx`와 `threadIdx`에 값이 할당된다. 이 좌표값을 이용해서 각 스레드들이 처리해야할 데이터를 할당해줄 수 있다.

> `blockIdx`와 `threadIdx`는 CUDA 내장 벡터 타입인 `uint3`이며 3차원을 표현한다. 각 컴포넌트는 멤버인 `x`, `y`, `z`로 접근할 수 있다.
>
> ex) `threadIdx.x`, `blockIdx.y`, `blockIdx.z`

CUDA에서는 그리드와 블록을 3차원으로 구성할 수 있다. 바로 위에서 본 그림은 2차원 블록을 포함하고 있는 2차원 그리드로 구성된 스레드 계층 구조를 표현하고 있다. 그리드와 블록의 차원 또한 커널 내에서 CUDA 내장 변수로 접근할 수 있다.

- `blockDim` : block dimension, measured in threads
- `gridDim` : grid dimension, measured in blocks

> `blockDim`과 `gridDim`은 `uint3` 기반의 내장 벡터 타입 `dim3` 타입이다. 이 타입을 사용할 때 지정되지 않은 차원의 값은 1을 기본값으로 가지게 된다. `uint3`과 마찬가지로 3차원으로 구성되며, 각 차원은 `x`, `y`, `z`로 접근할 수 있다.
>
> ex) `blockDim.x`, `blockDim.y`, `gridDim.z`

<br>

아래의 간단한 코드를 통해서 스레드와 블록, 그리고 그리드의 인덱스 및 차원이 어떻게 구성되는지 살펴볼 수 있다. 아래 코드에서는 처리할 데이터의 갯수를 6개로 지정했을 때, 블록과 그리드의 차원을 어떻게 지정하는지 보여준다. 결과적으로 아래 코드에서는 (3,1,1) 차원의 블록과 이들로 구성된 (2,1,1) 차원의 그리드를 생성한다.

> `main` 함수 마지막에 `cudaDeviceSynchronize()`를 호출하고 있다. 이 함수는 호출된 지점 이전에 실행 중이던 모든 GPU 연산이 종료될 때까지 기다리는 역할을 한다. 앞서 언급했듯이 GPU의 대부분의 연산은 비동기 동작으로 수행되기 때문에 커널이 실행되자마자 제어권은 다시 host로 전달된다. 환경마다 차이가 있을 수는 있지만, `cudaDeviceSynchronize()`가 없다면 대부분의 경우 커널이 실행되어 메세지를 출력하기 전에 `main` 함수가 리턴될 것이다.

```c++
#include <stdio.h>
#include <cuda.h>

__global__
void checkIndex()
{
    printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d)  blockDim: (%d, %d, %d)  gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char** argv)
{
    int num_elements = 6;

    // define grid and block structure
    dim3 block{3};  // (3, 1, 1)
    dim3 grid{(num_elements + block.x - 1) / block.x};  // (2, 1, 1)

    // check grid and block dimension from host side
    printf("grid.x %d  grid.y %d  grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d  block.y %d  block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // sync between host and device
    cudaDeviceSynchronize();

    return 0;
}
```

위 코드를 간단히 `nvcc`로 컴파일하고 실행하면 아래의 결과를 얻을 수 있다.

> `nvcc -o checkDim check_dim.cu` 로 컴파일했음

```
grid.x 2  grid.y 1  grid.z 1
block.x 3  block.y 1  block.z 1
threadIdx: (0, 0, 0) blockIdx: (0, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
threadIdx: (1, 0, 0) blockIdx: (0, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
threadIdx: (2, 0, 0) blockIdx: (0, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
threadIdx: (0, 0, 0) blockIdx: (1, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
threadIdx: (1, 0, 0) blockIdx: (1, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
threadIdx: (2, 0, 0) blockIdx: (1, 0, 0)  blockDim: (3, 1, 1)  gridDim:(2, 1, 1)
```

<br>

# CUDA Kernel

## How to launch a CUDA Kernel

CUDA 커널 호출은 **execution configuration** 이라는 문법을 통해 호출하며, `<<<grid, block>>>`을 함수 이름과 파라미터 리스트 사이에 위치시킨다.

```c++
kernel_name <<<grid, block>>> (argument list);
```

앞서 언급했듯, CUDA 프로그래밍 모델에는 스레드 계층이 있다. Execution configuration을 통해 스레드들이 어떻게 GPU에서 스케쥴링되는지 지정할 수 있는데, execution configuration의 첫 번째 값은 그리드 차원(실행할 블록의 수)이고 두 번째 값은 블록 차원(각 블록 내의 스레드 수)이다. 이를 통해, 커널을 실행할 전체 스레드 수와 스레드들의 레이아웃을 지정할 수 있다.

클러스터라는 개념은 일단 무시하면, 같은 블록 내의 스레드들은 서로 쉽게 커뮤니케이션이 가능하지만, 다른 블록에 위치한 스레드들은 서로 커뮤니케이션할 수 없다. 따라서, 문제를 어떻게 해결하느냐에 따라 그리드와 블록의 레이아웃을 다르게 하여 스레드를 구성할 수 있다. 예를 들어, 32개의 데이터 요소에 대해 연산을 수행할 때, 각 블록에 8개의 요소가 포함되도록 하여 아래와 같이 4개의 블록을 실행시킬 수 있으며, 스레드 레이아웃은 아래 그림과 같이 구성된다.

```c++
kernel_name<<<4, 8>>>(argument list);
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbecCyi%2FbtrYqzqIjS4%2FQdbqwlSU8baTecmaPaOQZ1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

Global memory 내에서 데이터는 선형적으로 저장되기 때문에 커널 내에서 내장 변수인 `blockIdx.x`와 `threadIdx.x`를 사용하면

- 그리드 내에서 고유한 스레드 식별
- 스레드와 데이터 요소 간의 매핑

이 가능하다.

> Thread와 data mapping에 대한 자세한 내용은 아래 포스트에서 조금 더 자세히 설명한다.
> 
> - [CUDA Thread 구조와 Data Mapping](https://junstar92.tistory.com/245)

만약 32개의 데이터를 하나의 블록에 모두 포함시킨다면, 다음과 같이 하나의 블록으로 커널을 실행시킬 수 있다.

```c++
kernel_name<<<1, 32>>>(argument list);
```

또는, 각 블록에 단 하나의 스레드만 포함하도록 한다면, 32개의 블록을 실행시킬 수도 있다.
```c++
kernel_name<<<32, 1>>>(argument list);
```

커널 함수는 host 스레드에 대해 비동기적(asynchronous)이다. 커널이 실행되면, 제어권은 곧바로 host 측으로 리턴된다. 만약 호출된 커널이 실행되는 모든 스레드가 완료될 때까지 강제로 host에서 기다리게 하려면, 아래의 함수를 호출하면 된다.

`cudaError_t cudaDeviceSynchronize(void);`

몇몇 CUDA Runtime API는 host와 device간의 암묵적인 동기화를 수행한다. 예를 들어, `cudaMemcpy`를 사용하여 host와 device간의 데이터 복사를 수행할 때, host 측에 대한 암묵적인 동기화가 수행되어 데이터 복사가 완료될 때까지 host가 기다린다. 

> 위에서 언급했듯이, 엄밀히 따지면 `cudaMemcpy`에서 host <-> device 간의 암묵적인 동기화가 항상 수행된다고 볼 수는 없다. 

`cudaMemcpy`는 이전의 모든 커널 호출이 완료된 이후에 복사를 수행한다.

## How to write a CUDA Kernel

커널 함수는 device 측에서 수행되는 코드이다. 커널 함수를 작성할 때, 한 스레드에서의 연산과 그 스레드에서 액세스되는 데이터를 정의한다. 그리고 커널이 호출될 때, 많은 CUDA 스레드들이 병렬적으로 동일한 연산을 수행하게 된다. 커널 함수는 `__global__` declaration specification를 사용하여 정의된다.

```c++
__global__ void kernel_name(argument list);
```

> 커널 함수의 리턴 타입은 반드시 `void`이어야 한다.

아래 표는 CUDA C 프로그래밍에서 사용 가능한 function type qualifiers를 보여준다. 이는 함수가 host 또는 device에서 실행되는지, host 또는 device에서 호출가능한 지에 대해서 알려준다.

|Qualifiers|Execution|Callable|Notes|
|--|--|--|--|
|`__global__`|Executed on the device|Callable from the host<br>Callable from the device for devices of compute capability 3| Must have a void return type|
|`__device__`|Executed on the device|Callable from the device only| |
|`__host__`|Executed on the host|Callable from the host only|Can be omitted|

`__device__`와 `__host__`는 함께 사용될 수 있으며, 이 경우에 함수는 host와 device에서 모두 컴파일된다.

간단한 예시로, 크기가 N인 두 벡터 A와 B를 더하는 코드를 host에서 작성하면 아래와 같을 것이다.
```c++
void vectorAddOnHost(float* A, float* B, float* C, const int N) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}
```

이를 커널 함수로 바꾸면 아래와 같이 작성할 수 있다.
```c++
__global__
void vectorAddOnGpu(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```

커널 함수는 일반 순차 코드와 다르게 반복문이 사라지고, 내장된 스레드 좌표 변수를 배열의 인덱스로 사용한 것을 볼 수 있다. 여기서는 간단하게 표현하기 위해서 배열의 access violation을 체크하지 않는다 (항상 N개의 스레드로 호출된다고 가정한다). 만약 32개의 요소를 갖는 배열에 대해서 위 커널 함수를 호출한다고 하면, 아래와 같이 32개의 스레드를 갖는 하나의 블록으로 호출할 수 있다.

```c++
vectorAddOnGpu<<<1, 32>>>(A, B, C);
```

<br>

# CUDA Error Handling

많은 CUDA 함수들은 비동기로 동작하기 때문에 에러가 발생한 루틴을 식별하기가 어렵다. 따라서, 일반적으로 아래와 같이 error-handling 매크로를 사용하여 CUDA API 호출을 래핑해 에러를 체크하는 프로세스를 적용한다.

```c++
#define CUDA_ERROR_CHECK(err) \
  if (err != cudaError_t::cudaSuccess) { \
      fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
      exit(err); \
  }
```

예를 들어, CUDA API를 호출할 때, 아래와 같이 매크로를 함께 사용한다.

```c++
CUDA_ERROR_CHECK(cudaMemcpy(dest, src, n_bytes, cudaMemcpyHostToDevice));
```

만약 메모리 복사나 이전에 실행된 비동기 연산이 에러를 일으켰다면, 위 매크로에서 에러 코드를 리포트하고 에러 메세지를 출력한 뒤, 프로그램을 종료시킨다. 이러한 에러 체크 프로세스는 아래와 같이 커널 실행에서도 적용 가능하다.

```c++
kernel_function<<<grid, block>>(argument_list);
CUDA_ERROR_CHECK(cudaDeviceSynchronize());
```

다만, 위 코드에서는 `cudaDeviceSynchronize()`가 이전에 요청된 태스크들이 완료할 때까지 host가 기다리게 되므로(global synchronization이 발생), 디버깅용으로만 사용하는 것이 좋다.

<br>

# Example: Vector Addition

> 예제에 사용된 전체 코드는 아래 링크 참조
> - Sequential Vector Addition: [vector_add_on_host.cpp](/code/cuda/vector_add/vector_add_on_host.cpp)
> - Parallel Vector Addition: [vector_add.cu](/code/cuda/vector_add/vector_add.cu) 

벡터 덧셈 예시를 가지고, sequential code와 CUDA로 구현한 parallel code를 살펴보자.

Sequential code에서 벡터 덧셈은 아래와 같이 구현할 수 있다.

```c++
void addVectorOnHost(float const* a, float const* b, float* c, const int num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i];
    }
}
```

필자의 PC에서 벡터의 요소 갯수를 $2^{20} = 1,048,576$ 개로 지정하여 실행시키면 약 2 ms의 시간이 걸린다고 측정된다. 출력 결과는 아래와 같다.
```
[Vector addition of 1048576 elements on Host]
Time: 2.079 msec
```

CUDA로 구현한 parallel vector addition code는 다음과 같다.

```c++
__global__
void vectorAddKernel(float const* a, float const* b, float* c, int const num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements)
        c[i] = a[i] + b[i];
}
```

다만, CUDA를 사용해서 데이터(벡터) 연산을 수행하려면 host memory에 있는 벡터값들을 device memory로 옮겨준 뒤, 커널을 실행시켜주어야 한다. 그리고, 연산 결과를 다시 host memory로 옮겨주는 과정이 필요하다. 이러한 일련의 프로세스를 수행하는 과정을 아래 `vectorAdd(...)` 함수에 구현하였다.

```c++
void vectorAdd(float const* a, float const* b, float* c, int const num_elements)
{
    // allocate the device input vectors a, b, c
    float *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, sizeof(float) * num_elements));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, sizeof(float) * num_elements));

    // copy the host input vector a and b in host memory
    // to the device input vectors in device memory
    printf("> Copy input data from the host memory to the CUDA device\n");
    CUDA_ERROR_CHECK(cudaMemcpy(d_a, a, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, b, sizeof(float) * num_elements, cudaMemcpyHostToDevice));

    // allocate CUDA events for estimating elapsed time
    cudaEvent_t start, stop;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // Launch the vectorAddKernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    printf("> CUDA kernel launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);

    CUDA_ERROR_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, num_elements);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));

    // copy the device result vector in device memory
    // to the host result vector in host memory
    printf("> Copy output data from the CUDA device to the host memory\n");
    CUDA_ERROR_CHECK(cudaMemcpy(c, d_c, sizeof(float) * num_elements, cudaMemcpyDeviceToHost));

    // verify that the result vector is correct
    printf("> Verifying vector addition...\n");
    for (int i = 0; i < num_elements; i++) {
        if (a[i] + b[i] != c[i]) {
            fprintf(stderr, "Result verification failed at element %d (%f != %f)\n", i, a[i]+b[i], c[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("> Test PASSED\n");

    // compute performance
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    double flops = static_cast<double>(num_elements);
    double giga_flops = (flops * 1.0e-9f) / (msec / 1000.f);
    printf("Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f ops, "
            " WorkgroundSize = %u threads/block\n",
            giga_flops, msec, flops, threads_per_block);
    
    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    CUDA_ERROR_CHECK(cudaFree(d_c));
    // free CUDA event
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
}
```

여기서는 CUDA API의 error handling을 위해 `CUDA_ERROR_CHECK` 매크로를 사용하여 에러를 체크해주고 있으며, 아직 언급하지는 않았지만 device에서 실행되는 커널의 수행 시간을 측정하기 위해 CUDA Event를 사용하고 있다. 커널의 수행 시간을 측정하거나 CUDA 프로그램을 프로파일링하는 방법에 대해서는 다른 포스팅을 통해 다루도록 한다.

그리고, 커널을 실행하기 전에 블록과 그리드의 차원을 결정해주고 있다. 여기서는 한 스레드 블록 당 256개의 스레드를 가지며 1차원 레이아웃으로 구성된다. 그리드의 차원은 전체 벡터 수에 따라 달라지는데, 나누어 떨어지지 않는 경우를 커버하기 위해 `(num_elements + threads_per_block - 1) / threads_per_block` 로 계산한다. 예를 들어, $2^{20}=1,048,576$ 개의 요소가 있다면, 그리드의 차원은 $(1,048,576 + 256 - 1) / 256 = 4096$ 이 된다. 마찬가지로 그리드도 1차원 레이아웃이다.

전체 코드는 이 챕터 시작 부분에 첨부한 링크를 참조바람.

이렇게 작성한 프로그램을 컴파일하고, 실행한 출력 결과는 아래와 같다. 사용된 GPU는 `RTX3080` 이다.
```
[Vector addition of 1048576 elements on GPU]
> Copy input data from the host memory to the CUDA device
> CUDA kernel launch with 4096 blocks of 256 threads
> Copy output data from the CUDA device to the host memory
> Verifying vector addition...
> Test PASSED
Performance = 36.01 GFlop/s, Time = 0.029 msec, Size = 1048576 ops,  WorkgroundSize = 256 threads/block
Done
```

벡터의 요소 수를 동일하게 설정했을 때, sequential code의 덧셈 수행 시간은 약 2ms 였지만, CUDA로 작성한 parallel code의 덧셈 수행 시간은 약 0.029ms로 약 69배 빨라졌다는 것을 확인할 수 있다. 이처럼 처리할 데이터가 아주 많고, 이를 병렬로 처리할 수 있다면 CUDA를 사용하는 것이 훨씬 빠르다는 것을 보여준다.

방금 언급했듯이, 데이터가 아주 많은 경우에 CUDA가 빠르다고 했는데, 극단적으로 데이터의 수를 256로 지정했을 때 그 결과가 어떻게 나오는지 보면 알 수 있다.

데이터의 수를 256개로 지정했을 때, sequential code의 결과는
```
[Vector addition of 256 elements on Host]
Time: 0.000 msec
```
이고, CUDA parallel code의 결과는
```
[Vector addition of 256 elements on GPU]
> Copy input data from the host memory to the CUDA device
> CUDA kernel launch with 1 blocks of 256 threads
> Copy output data from the CUDA device to the host memory
> Verifying vector addition...
> Test PASSED
Performance = 0.02 GFlop/s, Time = 0.011 msec, Size = 256 ops,  WorkgroundSize = 256 threads/block
Done
```

그 결과를 보면 알 수 있듯이, 데이터의 수가 상당히 적을 때에는 sequential code의 수행 시간이 더 빠르다는 것을 볼 수 있다. 이는 커널을 실행(launch)하고, 실제 device에서 이를 수행하는 오버헤드가 실제 코드 수행보다 더 크기 때문이다.

항상 병렬로 동작하는 것이 빠르지 않기 때문에 주의해야 한다.

이번에는 반대로 $2^{20}$ 보다 더 큰 $2^{25}$ 개로 지정하여 테스트한 결과를 비교해보자. Sequential code와 parallel code의 결과는 각각 아래와 같다.
```
[Vector addition of 33554432 elements on Host]
Time: 74.927 msec
```
```
[Vector addition of 33554432 elements on GPU]
> Copy input data from the host memory to the CUDA device
> CUDA kernel launch with 131072 blocks of 256 threads
> Copy output data from the CUDA device to the host memory
> Verifying vector addition...
> Test PASSED
Performance = 57.15 GFlop/s, Time = 0.587 msec, Size = 33554432 ops,  WorkgroundSize = 256 threads/block
Done
```

데이터의 수가 $2^{20}$ 일 때는 CUDA의 수행 시간이 CPU보다 약 69배 빨랐다. 하지만 데이터의 수가 32배가 된 경우, CUDA의 수행 시간은 CPU보다 약 127배 빠르다고 측정된다. 이처럼 데이터의 수가 많으면 많을수록 병렬로 처리하는 것이 더 빠르다.


<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documenation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)