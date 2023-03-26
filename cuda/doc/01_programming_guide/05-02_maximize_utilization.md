# Table of Contents

- [Table of Contents](#table-of-contents)
- [Maximize Utilization](#maximize-utilization)
  - [Application Level](#application-level)
  - [Device Level](#device-level)
  - [Multiprocessor Level](#multiprocessor-level)
  - [Occupancy Calculator](#occupancy-calculator)
- [References](#references)

<br>

# Maximize Utilization

GPU utilization을 최대로 끌어올리려면 가능한 많은 병렬 처리가 수행되도록 해야 하며, 이를 시스탬의 다양한 컴포넌트에 효율적으로 매핑하여 GPU가 바쁘게 동작하도록 어플리케이션을 구성해야 한다.

## Application Level

고수준의 관점에서 어플리케이션은 스트림과 비동기 함수 콜을 사용하여 host, device, 그리고 host와 device를 연결하는 bus 간의 parallel execution을 최대화해야 한다. 각 프로세서에는 해당 프로세서가 가장 잘 수행할 수 있는 타입의 작업을 할당해야 한다. 즉, host에는 serial workload, device에는 parallel workloads를 할당해야 한다.

Parallel workloads의 경우, 일부 스레드가 서로의 데이터를 공유하기 위해 동기화를 해야 하며 이때는 병렬 처리가 중단된다. 만약 데이터를 공유해야 하는 스레드들이 동일한 스레드 블록에 속한다면, 공유 메모리와 `__synchthreads()`를 사용하여 데이터를 공유해야 한다.

만약 서로 다른 스레드 블록에 속한다면, 두 개의 개별 커널(하나는 writing, 다른 하나는 reading)을 사용하여 global memory를 통해 데이터를 공유해야 한다. 이 경우에는 추가적인 커널 호출과 global memory traffic에 대한 오버헤드가 발생하기 때문에 첫 번째 방법에 비해 최적화되어 있지 않다.

따라서, 스레드 간 통신이 필요한 연산은 가능한 단일 스레드 블록 내에서 통신을 수행하는 방식의 알고리즘을 사용하여 global memory를 통한 데이터 공유는 최소화해야 한다.

## Device Level

저수준에서 device의 multiprocessors 간을 parallel execution을 최대화해야 한다. 여러 커널들은 device에서 동시에 실행될 수 있으며, 스트림을 사용하여 충분한 커널들이 동시에 실행될 수 있도록 하여 maximum utilization을 달성할 수 있다.

## Multiprocessor Level

저수준에서 multiprocessor 내의 다양한 `function units`간의 prallel execution을 최대화해야 한다. GPU multiprocessor는 기본적으로 thread-level parallelism을 통해 utilization을 극대화한다.

이러한 utilization은 상주하는 warps의 수에 직접적으로 연결된다. 모든 instruction이 **issue**될 때마다 warp scheduler는 실행할 준비가 된 instruction을 선택한다. 실행할 준비가 된 instruction이 선택되면 warp의 active threads에 issue 된다. Warp가 다음 instruction을 실행할 준비가 될 때까지 걸리는 clock cycles 수를 `latency`라고 부른다. 모든 warp schedulers가 latency 시간 동안 모든 clock cycle에서 일부 워프에 issue할 instruction을 항상 가지고 있을 때, full utilization을 달성하게 된다. 즉, latency가 숨겨져 드러나지 않는다는 것을 의미한다 (바로바로 instruction이 실행된다는 것을 의미).

`L` clock cycle의 latency를 hiding하는 데 필요한 instruction의 수는 instruction의 처리량에 따라 다르다 (다양한 산술 instruction의 처리량은 [Arithmetic Instruction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)에서 확인할 수 있다). 최대 처리량을 갖는 instruction이라고 가정한다면, latency를 hiding하는 데 필요한 instruction 수는 다음과 같다.

- `4L` for devices of compute capability 5.x, 6.1, 6.2, 7.x, 8.x. 이들 device에서 multiprocessor는 4개의 warp scheduler를 가지고 있어서, one clock cycle 동안 warp 당 하나의 instruction을 issue하기 때문이다.
- `2L` for devices of compute capability 6.0. 두 개의 서로 다른 warp에 대해 매 cycle마다 2개의 instruction가 issue되기 때문이다.

Warp가 다음 instruction을 실행될 준비가 되지 않는 가장 일반적인 원인은 instruction의 입력 피연산자(input operand)가 아직 사용할 수 없기 때문이다. 만약 모든 입력 피연산자가 레지스터(registers)라면, latency는 레지스터 종속성으로 인해 발생한다. 즉, 일부 입력 피연산자가 아직 완료되지 않은 이전 instruction에 의해 쓰여질 수 있다. 이 경우, latency는 이전 instruction의 실행 시간과 동일하며 warp schedular는 해당 latency 동안 다른 warp의 instruction으로 스케쥴링한다.

Compute capability 7.x인 GPU에서 대부분의 arithmetic instruction은 일반적으로 4 clock cycles를 가진다. 따라서, 이들 instruction의 latency를 hiding하려면 멀티프로세서당 16개의 active warps가 필요하다 (4 cycles, 4 warp schedulers). 만약 개별 warps들이 instruction-level parallelism으로 동작한다면, 즉, 여러 독립적인 instruction들이 instruction stream에 있는 경우, 단일 워프가 여러 독립적인 instruction을 연속적으로 실행할 수 있기 때문에 16개보다 더 적은 active warps로 충분하다.

일부 입력 피연산자가 off-chip memory에 상주한다면, latency는 훨씬 더 길어진다. 일반적으로 수백 clock cycles이다. 이렇게 긴 latency 동안 warp schedulers가 GPU를 바쁘게 만드는데 필요한 warp의 수는 커널 코드와 instruction-level parallelism 정도에 따라 다르다. 일반적으로 off-chip memory 피연산자가 없는 instruction 대비 off-chip memory 피연산자가 있는 instruction의 수가 적으면, 더 많은 warps가 필요하다 (이 비율은 일반적으로 `arithmetic intensity of the program`이라고 부른다).

Warp가 다음 instruction을 실행할 준비가 되지 않는 또 다른 이유는 일부 `memory fench` 또는 `synchronization point`에서 대기하고 있기 때문이다. 동기화 지점에서는 동일한 스레드 블록의 다른 warp의 실행이 완료될 때까지 대기하기 때문에 멀티프로세서를 강제로 idle 상태로 만들 수 있다. 멀티프로세서당 여러 개의 스레드 블록을 갖도록 하면, 다른 스레드 블록의 warp가 동기화 지점에서 서로를 기다릴 필요가 없기 때문에 idle을 줄이는 데 도움이 될 수 있다.

> 주어진 커널 호출에서 각 멀티프로세서에 상주하는 스레드 블록의 수는 execution configuration과 멀티프로세서의 메모리 리소스, 그리고 커널에서 필요한 리소스에 따라 다르다. 커널이 사용하는 레지스터 및 공유 메모리 사용량은 컴파일 할 때, `--ptxas-options=-v` 옵션을 추가하여 확인할 수 있다.

> 하나의 스레드 블록에서 필요한 공유 메모리의 총 크기는 정적으로 할당된 공유 메모리와 동적으로 할당된 공유 메모리의 합과 같다.

**커널이 사용하는 레지스터의 수는 상주하는 warp의 수에 상당한 영향을 미칠 수 있다.** 예를 들어, compute capability 6.x device에서 커널 함수가 64개의 레지스터를 사용하고 각 스레드 블록에는 512개의 스레드가 있고, 공유 메모리를 사용하지 않는다고 가정해보자. 그러면 두 개의 스레드 블록(즉, 32개의 warps = 1,024 threads)에는 `2x512x64=65536`개의 레지스터가 필요한데, 이는 멀티프로세서에서 사용 가능한 레지스터의 수와 동일하다. 여기서 이 커널이 레지스터 하나를 더 사용한다면, 두 개의 스레드 블록에서는 멀티프로세서에서 사용할 수 있는 레지스터보다 더 많은 양의 레지스터를 사용하게 되므로, 하나의 스레드 블록만이 멀티프로세서에 상주할 수 있게 된다. 이 경우, 컴파일러는 `register spilling`과 instruction의 수를 최소화하도록 하여, 레지스터의 사용량을 최소화하려고 시도한다.

> `register spilling`은 [Device Memory Access](/cuda/doc/01_programming_guide/03-05-03_maximize_memory_throughput.md)의 `local memory`에서 언급된다.
>
> 또한, [CUDA Memory Model: Registers](/cuda/study/09_cuda_memory_model.md#registers)에서도 따로 다루고 있다. 여기에는 레지스터의 수를 제한하는 방법도 언급하고 있다.

레지스터 파일(register file)dms 32-bit 레지스터로 구성된다. 따라서, 레지스터에 저장되는 각 변수는 적어도 하나의 32-bit 레지스터가 필요하다. 따라서, `double` 변수의 경우에는 2개의 레지스터를 사용하게 된다.

커널 호출에 대해 execution configuration이 성능에 미치는 영향은 일반적으로 커널 코드 구현에 따라 다르다. 따라서, 실험을 통해 최적의 execution configuration을 찾아야 한다. 어플리케이션에서는 GPU의 compute capability에 따라 달라지는 register file size/shared memory size를 기반으로 execution configuration을 파라미터화할 수 있다. 이러한 정보들은 `device query`를 통해 쿼리할 수 있다.

Execution configuration을 설정할 때, 한 가지 중요한 점은 리소스를 낭비하지 않도록 블록 당 스레드 수를 warp 크기의 배수로 선택하는 것이 좋다는 것이다.

## Occupancy Calculator

몇몇 런타임 API를 통해 주어진 커널이 사용하는 register와 shared memory를 기반으로 스레드 블록의 크기와 클러스터의 크기를 선택할 수 있다.

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`
  
  블록 사이즈와 커널의 shared memory 사용량을 기반으로 occupancy prediction을 제공한다. 이 함수는 멀티프로세서당 동시에 실행되는 스레드 블록의 수 기반으로 occupancy를 리포트한다. 반환된 값에 블록 당 warp의 수를 곱해주면 멀티프로세서당 동시에 실행되는 warp의 수를 계산할 수 있고, 이를 멀티프로세서당 동시에 실행할 수 있는 최대 warp의 수로 나누어주면 occupancy를 백분율로 표현할 수 있다.
- `cudaOccupancyMaxPotentialBlockSize()` & `cudaOccupancyMaxPotentialBlockSizeVariableSMem()`
  
  휴리스틱으로 최대 수준의 멀티프로세서 occupancy를 달성할 수 있는 execution configuration을 계산한다.
- `cudaOccupancyMaxActiveClusters()`
  
  클러스터의 크기, 블록 크기, 커널의 shared memory 사용량을 기반으로 occupancy prediction을 제공한다.

아래 예제 코드는 간단한 `MyKernel`의 occupancy를 `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`를 사용하여 계산하는 방법을 보여준다. 여기서 결과는 동시에 실행되는 warp의 수와 멀티프로세서에서의 최대 warps의 수의 비율로 occupancy level을 나타낸다.

```c++
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    return 0;
}
```

위 코드 실행 결과는 다음과 같다 (`RTX 3080` 기준).
```
Occupancy: 33.3333%
```

아래 예제 코드는 `cudaOccupancyMaxPotentialBlockSize()`를 사용하여 maximum occupancy를 달성하기 위한 최소한의 grid 크기와 block의 크기를 계산하는 방법을 보여준다.

```c++
#include <iostream>

// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size
    
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);
    printf("Calculated minGridSize: %d / blockSize: %d\n", minGridSize, blockSize);
    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    printf("Launch MyKernel<<< %4d, %4d >>>\n", gridSize, blockSize);
    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}

int main()
{
    int num_elements = 1 << 22;

    int* h_arr = (int*)malloc(num_elements * sizeof(int));
    for (int i = 0; i < num_elements; i++) {
        h_arr[i] = rand() & 0xFF;
    }

    int* d_arr;
    cudaMalloc(&d_arr, num_elements * sizeof(int));
    cudaMemcpy(d_arr, h_arr, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    launchMyKernel(d_arr, num_elements);
    
    cudaMemcpy(h_arr, d_arr, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
```

위 코드를 실행한 결과는 다음과 같다. `cudaOccupancyMaxPotentialBlockSize()`를 통해 얻은 블록 크기를 통해 execution configuration을 설정했다.
```
Calculated minGridSize: 136 / blockSize: 768
Launch MyKernel<<< 5462,  768 >>>
```

<br>

# References

- [NVIDIA CUDA Documentations: Maximize Utilization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-utilization)