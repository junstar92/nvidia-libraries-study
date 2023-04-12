# Table of Contents

- [Table of Contents](#table-of-contents)
- [Device Memory Spaces](#device-memory-spaces)
- [Coalesced Access to Global Memory](#coalesced-access-to-global-memory)
  - [A Simple Access Pattern](#a-simple-access-pattern)
  - [A Sequential but Misaligned Access Pattern](#a-sequential-but-misaligned-access-pattern)
  - [Effects of Misaligned Accesses](#effects-of-misaligned-accesses)
  - [Stride Accesses](#stride-accesses)
- [L2 Cache](#l2-cache)
  - [L2 Cache Access Window](#l2-cache-access-window)
  - [Tuning the Access Window Hit-Ratio](#tuning-the-access-window-hit-ratio)
- [Shared Memory](#shared-memory)
  - [Shared Memory and Memory Banks](#shared-memory-and-memory-banks)
  - [Shared Memory in Matrix Multiplication (C=AB)](#shared-memory-in-matrix-multiplication-cab)
  - [Shared Memory in Matrix Multiplication (C=AAT)](#shared-memory-in-matrix-multiplication-caat)
  - [Asynchronous Copy from Global Memory to Shared Memory](#asynchronous-copy-from-global-memory-to-shared-memory)
- [Local Memory](#local-memory)
- [Texture Memory](#texture-memory)
  - [Additional Texture Capabilities](#additional-texture-capabilities)
- [Constant Memory](#constant-memory)
- [Registers](#registers)
  - [Register Pressure](#register-pressure)
- [References](#references)

<br>

# Device Memory Spaces

CUDA device에는 서로 다른 특징을 갖는 다양한 메모리 공간이 있는데, 아래 그림처럼 global, local, shared, texture, registers가 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/memory-spaces-on-cuda-device.png" width=500px style="display: block; margin: 0 auto; background-color: white"/>

이러한 메모리 공간 중에서 global memory가 가장 크다. Global memory, local memory, texture memory는 access latency가 가장 길고, 그 다음으로는 constant memory, shared memory, register file 순이다.

각 메모리 타입의 특징은 [Table 1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces__salient-features-device-memory)에서 확인할 수 있다.

<br>

# Coalesced Access to Global Memory

CUDA-capable GPU 아키텍처 프로그래밍에서 최상의 성능을 위해서는 global memory access를 병합(coalesced)해야 한다. Warp 내 스레드에 의한 global memory loads/stores는 coalesced global memory access를 통해 가능한 한 적은 memory transaction으로 처리될 수 있다. 가능한 한 global memory는 coalesced access로 접근해야 한다.

> Global memory에 대한 coalesced access requirement는 compute capability에 따라 다르다.

Compute capability 6.0 이상의 device에서 global memory의 coalesced access requirement는 꽤 간단하다. Warp 내 스레드들에 의한 concurrent access는 필요한 32-byte transactions의 수와 동일한 수의 transaction으로 병합된다.

Compute capability 5.2의 특정 device는 global memory에 대한 액세스의 L1 캐싱이 선택적으로 활성화될 수 있다. 이러한 장치에서 L1 캐싱이 활성화된 경우에는 128-byte memory transaction으로 처리된다. 따라서, 필요한 transaction의 수는 128-byte aligned segments의 수와 동일하다.

> Compute capability 6.0 이상의 device에서는 L1 캐싱이 default이지만, L1 캐싱의 활성화 여부와 상관없이 데이터 액세스 단위는 32-byte 이다.

> GDDR 메모리가 있는 device에서는 ECC 켜져 있을 때, coalesced access가 훨씬 더 중요하다. Scattered access는 ECC memory transfer 오버헤드를 증가시키는데, 특히, global memory에 데이터를 쓸 때 크게 증가시킨다.

아래 하위 섹션에서는 coalesced access에 대한 개념들을 예시롤 통해 설명한다. 여기서는 compute capability 6.0이라고 가정하며, 별 달리 언급되지 않는 한 스레드에서의 엑세스는 4-byte word라고 가정한다.

## A Simple Access Pattern

먼저 k번째 스레드가 32-byte로 정렬되어 있는 배열의 k번째 word에 액세스하는 간단한 액세스 패턴을 살펴보자. 여기서 모든 스레드가 참여할 필요는 없다.

예를 들어, warp 내 스레드들이 인접한 4-byte words에 액세스한다면 4개의 coalesced 32-byte transactions로 스레드들의 메모리 액세스를 처리할 수 있다. 이러한 패턴을 그림으로 나타내면 아래와 같다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/coalesced-access.png" width=400px style="display: block; margin: 0 auto; background-color: white"/>

이러한 패턴은 4개의 32-byte transactions를 발생시키며, 위 그림에서는 빨간색 사각형으로 나타내고 있다.

4개의 32-byte 세그먼트 중 하나의 세그먼트만 요청되는 경우, 예를 들어, 여러 스레드가 동일한 word에 액세스하거나 일부 스레드가 액세스에 참여하지 않는 경우에도 어쨌든 4개의 세그먼트를 페치한다.

> 하나의 32-byte 세그먼트만 사용하도록 커널을 실행하여 `nsight compute`로 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 메트릭을 측정해보면, 4 sector로 측정될 것이라고 예상했지만 1 sector로 측정된다. 실제로 1 sector만 처리되는 것인지 확인이 필요한 부분이다.

## A Sequential but Misaligned Access Pattern

만약 warp 내 연속된 스레드들이 연속적이지만 32-byte 세그먼트로 정렬되지 않은 메모리에 액세스하는 경우에는 아래 그림과 같이 5개의 32-byte 세그먼트가 요청된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/misaligned-sequential-addresses.png" width=400px style="display: block; margin: 0 auto; background-color: white"/>

`cudaMalloc()`와 같은 CUDA Runtime API를 통해 할당된 메모리는 최소 256 bytes로 정렬되는 것이 보장된다. 따라서, warp 크기(`32`)의 배수로 스레드 블록의 크기를 설정하면 warp들에서 메모리 액세스가 용이하다.

## Effects of Misaligned Accesses

Misaligned access에 의한 효과는 아래의 간단한 복사 커널을 통해 살펴볼 수 있다.

```c++
__global__
void offsetCopy(float *odata, float* idata, int offset)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    odata[xid] = idata[xid];
}
```

`offsetCopy` 커널은 input array `idata`로부터 output array로 데이터를 복사하며, 두 배열은 모두 global memory에 위치한다. `offset`을 0에서 32까지로 설정하여 실행하여 측정한 effective bandwidth 결과는 다음과 같다 (Tesla V100에서 측정한 결과).

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/performance-of-offsetcopy-kernel.png" width=600px style="display: block; margin: 0 auto; background-color: white"/>

Tesla V100(compute capability 7.0)에서 오프셋이 없거나 오프셋이 8의 배수인 경우(정확히는 8 words의 배수), 4개의 32-byte transactions이 요청된다. 측정된 bandwidth는 약 790GB/s 이다. 이외의 오프셋에서는 warp 당 5개의 32-byte transactions이 요청될 것이고, 이때 bandwidth는 대략 오프셋이 없을 때의 4/5가 될 것이라고 예상할 수 있다.

하지만, 이 결과에서 오프셋(8의 배수 이외)이 있을 때 달성한 메모리 처리량은 약 9/10이다. 이는 인접한 warp에서 페치한 **캐시 라인을 재사용**하기 때문이다. 따라서, misaligned access에 의한 영향을 있지만 예상했던 만큼 크지는 않다. 만약 인접한 warp로부터 페치된 캐시 라인을 재사용하지 않았다면 영향은 더 컸을 것이다.

> [Example: Misaligned Reads](/cuda/study/11_memory_access_patterns.md#example-misaligned-reads)에서 위에서 설명한 `offsetCopy` 커널과 유사한 커널로 테스트한 결과가 있다.

## Stride Accesses

바로 위에서 확인할 수 있듯이 misaligned sequential accesses의 경우, 캐시로 인해 성능에 미치는 영향을 줄일 수 있다. 하지만, non-unit-strided accesses 패턴에서는 조금 다를 수 있는데, 이러한 패턴은 주로 다차원 데이터 또는 행렬을 처리할 때 자주 발생한다. 이러한 패턴에서는 캐시 라인에 페치된 데이터를 최대한 많이 사용하도록 보장하는 것이 성능 최적화에서 중요하다.

Strided access가 성능에 미치는 영향은 아래 커널 `strideCopy()`를 통해 확인할 수 있다.
```c++
__global__ void strideCopy(float *odata, float* idata, int stride)
{
    int xid = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
    odata[xid] = idata[xid];
}
```

아래 그림은 warp 내에서 stride의 크기가 2인 strided access 패턴을 보여준다. 이제 더 이상 인접한 스레드들은 서로 인접한 데이터에 액세스하지 않고 하나 떨어진 위치의 데이터에 액세스하게 된다 (정렬되었다고 가정). 따라서, 기존에는 4개의 32-byte transactions으로 처리되었지만, 이제는 8개의 32-byte transactions로 처리된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/adjacent-threads-accessing-memory-with-stride-of-2.png" width=300px style="display: block; margin: 0 auto; background-color: white"/>

Stride가 2인 경우에는 50%의 load/store 효율을 보여준다 (요청된 메모리에서 절반의 요소에만 액세스하므로). 따라서, bandwidth가 낭비되고 있다. Stride가 증가하면 warp 내 32개의 스레드에 대해서 32개의 memory transactions로 처리되는 지점까지 bandwidth가 감소하게 된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/performance-of-stridecopy-kernel.png" width=650px style="display: block; margin: 0 auto; background-color: white"/>

위 지표에서 볼 수 있듯이 non-unit-stride global memory accesses는 가능한 피해야 한다. 이를 위한 한 가지 방법은 아래에서 다룰 [shared memory](#shared-memory)를 사용하는 것이다.

> Global memory access와 관련한 내용은 아래 포스팅에서 조금 더 자세히 다룬다.
> - [Global Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)
> - [Matrix Transpose Problem](/cuda/study/11-1_matrix_transpose_problem.md)

<br>

# L2 Cache

L2 캐시는 on-chip이므로 잠재적으로 global memory에 대해서 더 높은 bandwidth와 더 낮은 latency의 액세스를 제공할 수 있다. CUDA 11.0부터 compute capability 8.0 이상의 device에서는 L2 캐시의 데이터 지속성에 영향을 줄 수 있다.

> 이에 대한 자세한 내용은 문서의 [Device Memory L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management)에서 다루고 있다.

## L2 Cache Access Window

CUDA 커널이 global memory의 데이터 영역에 반복적으로 액세스할 때, 이러한 액세스는 지속되는 것으로 간주할 수 있다. 반면, 데이터가 오직 한 번만 액세스되는 경우에는 이러한 액세스를 스트리밍(streaming)으로 간주할 수 있다. L2 캐시 일부는 global memory 데이터 영역의 지속적인 액세스(persistent accesses)를 위해 별도로 설정할 수 있다.

지속적인 액세스로 설정되는 L2 캐시의 크기는 제한된 크기 내에서 조절될 수 있다.
```c++
cudaGetDeviceProperties(&prop, device_id);
/* Set aside max possible size of L2 cache for persisting accesses */
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);
```

CUDA 스트림 또는 CUDA 그래프 커널 노드의 access policy window를 사용하여 user data를 L2 일부로 매핑할 수 있는데, 아래 예제 코드는 CUDA 스트림에서 access policy window를 사용하는 방법을 보여준다.
```c++
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

## Tuning the Access Window Hit-Ratio

`num_bytes` 파라미터의 값과 L2 캐시의 크기에 따라서 L2 캐시 라인의 thrashing을 피하기 위해 `hitRatio` 값을 조정해야할 필요가 있다. `hitRatio` 파라미터는 `hitProp` 속성을 받는 액세스의 비율을 지정하는데 사용될 수 있다. 예를 들어, `hitRatio`의 값이 0.6이라면, global memory 영역 [ptr...ptr+num_bytes) 내에서 메모리 액세스의 60%는 persisting 속성을 가지고 메모리 액세스의 40%는 streaming 속성을 갖게 된다.

Sliding window microbenchmark를 사용하면, `hitRatio`와 `num_bytes`의 영향을 이해할 수 있다. Microbenchmark에서는 1024 MB의 GPU global memory를 사용한다. 그리고, persisting accesses를 위한 L2 캐시를 `cudaDeviceSetLimit()`을 사용하여 30 MB로 할당한다. 그런 다음, 아래 그림과 같이 메모리 영역의 첫 번째 `freqSize * sizeof(int)` 바이트에 대한 액세스를 persistent로 지정한다. 따라서, 이 데이터는 L2 캐시에서 사용된다. 이 benchmark에서는 이러한 persistent data 영역의 크기를 10 MB에서 60 MB까지 변화시켜 다양항 시나리오를 모델링한다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/sliding-window-l2.png" width=700px style="display: block; margin: 0 auto; background-color: white"/>

Tesla A100 GPU의 L2 캐시 용량은 총 40 MB이므로 이에 주의해야 한다. 지정한 L2 캐시 영역 외 데이터 액세스(즉, streaming data)는 normal 또는 streaming 액세스로 간주되며 지정되며 L2 캐시로 지정된 영역 중 사용되지 않는 부분이 있다면 나머지 L2 캐시 영역을 사용한다. 즉, 위 예시에서는 30 MB 외 나머지 10 MB가 이에 해당한다.

```c++
__global__ void kernel(int *data_persistent, int *data_streaming, int dataSize, int freqSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /*Each CUDA thread accesses one element in the persistent data section
      and one element in the streaming data section.
      Because the size of the persistent memory region (freqSize * sizeof(int) bytes) is much
      smaller than the size of the streaming memory region (dataSize * sizeof(int) bytes), data
      in the persistent region is accessed more frequently*/

    data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize];
    data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
}

stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);   //Number of bytes for persisting accesses in range 10-60 MB
stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                      //Hint for cache hit ratio. Fixed value 1.0
```

위 커널의 성능은 아래에서 확인할 수 있다. Persistent data 영역이 L2 캐시로 할당한 30 MB에 잘 맞으면 성능이 최대 50% 정도 향상된다. 하지만, persistent data 영역의 크기가 L2 캐시 크기를 초과하면 L2 캐시라인의 trashing으로 인해 약 10%의 성능 저하가 발생하는 것을 관측할 수 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/l2-hitratio-before.png" width=800px style="display: block; margin: 0 auto; background-color: white"/>

Persistent 데이터의 크기가 설정된 L2 캐시 영역의 크기보다 큰 경우에 성능을 최적화하기 위해서 access window의 `num_bytes`와 `hitRatio` 파라미터를 아래와 같이 조정하면 된다.
```c++
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
stream_attribute.accessPolicyWindow.num_bytes = 20*1024*1024;                                  //20 MB
stream_attribute.accessPolicyWindow.hitRatio  = (20*1024*1024)/((float)freqSize*sizeof(int));  //Such that up to 20MB of data is resident.
```
위 코드에서 access window의 `num_bytes`를 20 MB로 고정시키고 persistent data 중 임의의 20 MB 크기의 데이터가 별도로 설정된 L2 캐시에 상주하도록 `hitRatio`를 조정한다. Persistent 속성으로 지정되지 않은 persistent data의 나머지 부분은 streaming 속성으로 액세스된다. 이렇게 하면 캐시의 trashing을 줄이는 데 도움이 된다. 이렇게 설정하여 측정한 결과는 아래와 같으며, persistent data의 크기가 설정한 L2 캐시 영역의 크기에 일치하는지 여부와 상관없이 좋은 성능을 관측할 수 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/l2-hitratio-after.png" width=800px style="display: block; margin: 0 auto; background-color: white"/>

<br>

# Shared Memory

Shared memory는 on-chip 메모리이므로 locam memory와 global memory보다 훨씬 높은 bandwidth와 낮은 latency를 가진다 (bank conflict가 없는 경우).

> Shared memory에 대한 자세한 내용은 아래 포스팅에서도 다루고 있다.
> - [Shared Memory](/cuda/study/12_shared_memory.md)
> - [Data Layout of Shared Memory](/cuda/study/12-1_data_layout_of_shared_memory.md)
> - [Reducing Global Memory Access](/cuda/study/12-2_reducing_global_memory_access.md)
> - [Coalescing Global Memory Accesses](/cuda/study/12-3_coalescing_global_memory_accesses.md)
> - [Example: Matrix Multiplication](/cuda/doc/01_programming_guide/03-02-04_shared_memory.md)

## Shared Memory and Memory Banks

Shared memory는 동시에 액세스할 수 있는 동일한 크기의 메모리 모듈(**banks**)로 분할된다. 따라서, n개의 서로 다른 메모리 뱅크에 있는 n개의 주소를 읽거나 쓰는 경우, 이를 동시에 처리하여 단일 메모리 뱅크대비 n배 높은 effective bandwidth를 달성할 수 있다.

그러나 하나의 메모리 요청에서 여러 주소가 동일한 메모리 뱅크에 매핑되는 경우(bank conflict), 액세스가 동시가 아닌 순차적으로 이루어진다. 하드웨어는 bank conflicts가 있는 메모리 요청을 뱅크가 없는 별도의 요청으로 분리하여 처리하며, 별도로 추가되는 요청만큼 effective bandwidth가 감소한다. 한 가지 예외는 warp 내 여러 스레드들이 동일한 shared memory 위치에 액세스하는 경우인데, 이 경우에는 broadcast가 발생하여 요청된 shared memory 위치에서 스레드로의 단일 multicast로 병합된다.

Bank conflict를 최소화하려면, 메모리의 주소가 메모리 뱅크에 매핑되는 방법과 메모리 요청을 최적으로 스케줄링하는 것이 중요하다.

Compute capability 5.0 이상의 device인 경우, 각 뱅크로 매 클럭 주기마다 32비트의 bandwidth를 가지며, 연속하는 32비트 word는 연속적인 뱅크에 할당된다. Warp의 크기는 32 threads이고 뱅크 수도 32이므로 bank conflict는 모든 스레드들 사이에서 발생할 수 있다.

> 자세한 내용은 위에 나열한 포스팅이나 문서의 [Compute Capability 5.x: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)를 참조바람

## Shared Memory in Matrix Multiplication (C=AB)

> 행렬 곱셈 전체 코드는 [matmul.cu](/cuda/code/matmul/matmul.cu)이나 샘플 코드 [matrixMul](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul)을 참조 바람

Shared memory를 사용하면 블록 내 스레드들이 서로 협업할 수 있다. 블록 내 여러 스레드들이 global memory의 동일한 데이터를 사용할 때, 공유 메모리를 사용하면 global memory의 데이터를 한 번만 액세스하면 된다. Coalesced access 패턴으로 데이터를 읽고 shared memory에 저장하고, 그런 다음 shared memory에서 재정렬(reorder)하여 uncoalesced memory accesses를 피할 수 있다. Bank conflict를 제외하면, shared memory에서는 non-sequential 또는 unaligned accesses에 대한 패널티는 없다.

간단한 행렬 곱셈 `C=AB`를 통해 shared memory의 사용을 설명할 수 있다. 여기서 `A` 행렬의 차원은 `Mxw`, `B` 행렬의 차원은 `wxN`, `C` 행렬의 차원은 `MxN`이다. 아래에서 살펴볼 행렬 곱셈 커널은 간단히 구현하기 위해 `M`과 `N`의 값은 32의 배수로 설정한다.

이 문제는 `wxw` 크기의 스레드 블록 및 타일을 사용하여 풀 수 있다. 타일 관점에서 `A`는 열 행렬, `B`는 행 행렬이며 `C`는 이들의 외적이다. `A`와 `B`의 각 타일을 계산하면 `C` 행렬의 일부분을 계산할 수 있다. 그림으로 표현하면 아래와 같다. `(N/w, M/w)` 블록 차원의 그리드가 실행되며, 각 스레드 블록은 주어진 타일에서 행렬 곱셈을 담당하게 된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/matrix-multiplication-block-column-by-block-row.png" width=300px style="display: block; margin: 0 auto; background-color: white"/>

이를 아래의 `simpleMultiply` 커널로 구현할 수 있다. 이 커널은 최적화가 전혀되지 않은 naive matrix multiplication 구현이다.
```c++
// Unoptimized matrix multiplication
__global__
void simpleMultiply(float *a, float* b, float *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * b[i*N+col];
    }
    c[row*N+col] = sum;
}
```
위 커널 함수의 파라미터 `a`, `b`, `c`는 행렬 A, B, C에 대한 global memory를 가리키는 포인터이다. 여기서 `blockDim.x`, `blockDim.y`, `TILE_DIM`은 모두 `w`로 동일하다. `wxw` 스레드 블록의 각 스레드들은 C의 타일의 한 요소를 계산한다.

Tesla V100에서 이 커널의 effective bandwidth는 119.9 GB/s이다. 성능을 분석해보려면, for 루프에서 global memory에 액세스하는 패턴을 살펴봐야 한다. 아래 그림과 같이 각 warp의 스레드들은 `A`의 하나의 행과 `B`의 전체 타일에 대해 `C`의 타일의 하나의 행을 계산하는 것을 볼 수 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/computing-row-of-tile.png" width=300px style="display: block; margin: 0 auto; background-color: white"/>

여기서 warp 내 모든 스레드들은 행렬 A에 대해 global memory로부터 동일한 값을 읽는다. 비록 이러한 액세스가 루프의 반복을 통해 캐시라인을 재사용할 수 있지만, 수 많은 warp가 동일한 멀티프로세서에서 동시에 실행되면 캐시라인이 쉽게 유지되지 않는다. 따라서 bandwidth의 낭비가 발생하게 된다.

행렬 곱셈에서 global memory load 효율을 개선하기 위해 shared memory를 사용하면 성능을 개선할 수 있다.
```c++
// Using shared memory to improve the global memory load efficiency in matrix multiplication
__global__
void coalescedMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    __syncwarp();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* b[i*N+col];
    }
    c[row*N+col] = sum;
}
```
`coalescedMultiply` 커널에서 `A`의 타일의 각 요소는 global memory로부터 coalesced access 패턴으로 통해 shared memory로 딱 한 번만 읽는다. for 루프의 각 반복에서 shared memory의 값은 warp 내 모든 스레드들로 브로드캐스트된다. 딱 한 번만 global memory로부터 읽고, 이후에는 shared memory에 저장된 값을 읽기 때문에 기존보다 더 성능이 좋다. Tesla V100에서 이 커널의 effiective bandwidth는 144.4 GB/s이다. 이는 L1 캐시 이용률이 좋지 않을 때, shared memory를 캐시처럼 사용하면 더 좋은 성능을 얻을 수 있다는 것을 보여준다.

추가적인 최적화를 위해 행렬 B도 shared memory를 이용하여 처리할 수 있다. 원리는 동일하다.
```c++
// Improvement by reading additional data into shared memory
__global__
void sharedABMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}
```
위 커널의 경우, Tesla V100에서의 effective bandwidth는 195.5 GB/s 이다. 이러한 성능 개선의 이유는 global memory로부터 중복 사용을 피했기 때문이다.

각 구현의 effective bandwidth는 Tesla V100에서 아래와 같이 측정된다고 한다.
|Optimization|Effective Bandwidth|
|--|--|
|No optimization|119.9 GB/s|
|Coalesced using shared memory to store a tile of A|144.4 GB/s|
|Using shared memory to eliminate redundant reads of a tile of B|195.5 GB/s|

## Shared Memory in Matrix Multiplication (C=AAT)

이번에는 $C=AA^\top$ 의 행렬 곱셈을 통해서 global memory에 대한 strided access와 shared memory bank conflicts를 어떻게 처리하는지 살펴본다.

먼저 이 행렬 곱셈의 naive 구현은 다음과 같다.
```c++
// Unoptimized handling of strided accesses to global memory
__global__
void simpleMultiply(float *a, float *c, int M)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
    }
    c[row*M+col] = sum;
}
```
위 코드에서 C 행렬의 *row* 행, *col* 열의 요소는 A의 *row* 번째 행과 *col* 번째 행의 내적으로 계산한다. 위 커널의 effective bandwidth는 Tesla V100에서 12.8 GB/s이다. 이는 naive 버전의 `C=AB` 행렬 곱셈 커널보다 훨씬 낮은 수치이다. 차이점은 for 루프 내의 식의 두 번째 항인 `a[col*TILE_DIM+i]` 이다. Warp 내 스레드에서 `col`은 전치된 A 행렬의 연속적인 열을 나타낸다. 그러므로, `col*TILE_DIM`으로 인해 stride가 `w`인 strided access 패턴으로 global memory를 액세스하게 되어 bandwidth를 낭비하게 된다.

> 문서 내에서 언급되지 않았지만, 

이전 섹션에서 봤듯이, shared memory를 사용하여 strided access를 해결할 수 있다. 아래 커널이 바로 shared memory를 사용하여 strided access를 해결한 최적화 커널이며, 여기서 warp는 행렬 A의 행을 shared memory 타일의 열로 저장한다.
```c++
// An optimized handling of strided accesses using coalesced reads from global memory
__global__
void coalescedMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}
```
위 커널은 `transposedTile`이라는 shared memory를 사용하여 for 루프 내 식의 두 번째 항에서 발생하는 uncoalesced access 문제를 해결한다. 또한, `aTile`을 사용하여 이전 섹션에서 적용한 중복된 global memory access를 단 한 번만 액세스하도록 한다. 위 커널의 effective bandwidth는 Tesla V100에서 140.2 GB/s로 측정된다. 그래서 이전 섹션에서 달성한 195.5 GB/s에 많이 미치지 못하는 결과이다. 이러한 결과의 차이는 **shared memory bank conflict** 때문이다.

루프 내에서 `transposedTile`의 각 요소를 읽는 데에는 bank conflict가 발생하지 않는다. 이는 각 warp의 스레드들이 타일의 열을 읽기 때문이다. 그러나 global memory로부터 `transposedTile`에 값을 복사할 때 bank conflict가 발생한다. Shared memory의 열에 global memory로부터 읽은 값을 쓰는데, `wxw` 타일을 사용하기 때문에 매 스레드마다 `w` bank 만큼의 stride가 발생한다. 따라서, 모든 스레드들은 동일한 bank의 메모리를 요청하게 되며, 이러한 many-way bank conflict는 처리하는데 비용이 매우 크다. 이를 해결하려면, 다음과 같이 shared memory 배열에 padding을 추가하면 된다.
```c++
__shared__ float transposedTile[TILE_DIM][TILE_DIM+1];
```
이렇게 padding을 추가해주면, bank conflict가 완전히 제거된다. 스레드 간 stride는 이제 `w+1` bank가 되어 모듈러 연산으로 unit(1) stride bank와 동일하게 처리된다. 이렇게 적용한 커널의 effective bandwidth는 Tesla V100에서 199.4 GB/s로 축정된다.

|Optimization|Effective Bandwidth|
|--|--|
|No optimization|12.8 GB/s|
|Using shared memory to coalesce global reads|140.2 GB/s|
|Removing bank conflicts|199.4 GB/s|

행렬 곱 연산 커널 최적화 섹션을 통해 shared memory를 사용하는 이유를 정리하면 다음과 같다.

- To enable coalesced accesses to global memory, especially to avoid large strides (for general matrices, strides are much larger then 32)
- To eliminate (or reduce) redundant loads from global memory
- To avoid wasted bandwidth

## Asynchronous Copy from Global Memory to Shared Memory

CUDA 11.0에서는 device code 내에서 사용할 수 있는 *async-copy* 기능이 도입되었다. 이를 사용하면 global memory로부터 shared memory로의 비동기 복사를 관리할 수 있다. 이 기능을 활용하면 CUDA 커널이 global to shared memory copy와 computation을 오버랩시킬 수 있다. 또한, global memory read와 shared memory write 간에 일반적으로 존재하는 중간 레지스터 파일에 대한 액세스를 피할 수 있다.

> 자세한 내용은 [`memcpy_async` API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memcpy-async-api)를 참조

Global to shared memory의 동기 및 비동기 복사의 성능 차이를 확인하기 위해 아래의 benchmark CUDA 커널을 살펴보자.
```c++
template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    uint64_t clock_start = clock64();

    for (size_t i = 0; i < copy_count; ++i) {
        shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
    }

    uint64_t clock_end = clock64();

    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    uint64_t clock_start = clock64();

    //pipeline pipe;
    for (size_t i = 0; i < copy_count; ++i) {
        __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                &global[blockDim.x * i + threadIdx.x], sizeof(T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    uint64_t clock_end = clock64();

    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}
```
동기 버전의 커널은 global memory의 값을 중간 레지스터(intermediate register)로 로드한 뒤, 레지스터의 값을 shared memory로 저장한다. 반면, 비동기 버전의 커널에서는 global memory의 값을 shared memory로 직접 저장하는 instruction이 `__pipeline_memcpy_async()` 함수가 호출되자마자 발생한다. `__pipeline_wait_prior(0)`은 pipe object 내의 모든 instruction이 실행될 때까지 대기한다. 비동기 복사를 사용하면 어떠한 중간 레지스터도 사용하지 않는다. 이를 사용하면 register pressure를 줄일 수 있고, 이를 통해 kernel occupancy를 증가시킬 수 있다. 비동기 복사 명령어를 사용한 global to shared memory 복사는 L1 캐시에 캐싱될 수 있고, 선택적으로 L1 캐시를 패싱할 수도 있다. 만약 각 CUDA 스레드가 16바이트의 요소들을 복사한다면, L1 캐시는 패싱된다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/sync-vs-async.png" width=700px style="display: block; margin: 0 auto; background-color: white"/>

위의 커널에 대해서 템플릿 파라미터는 `int`(4B), `int2`(8B), `int4`(16B)로 각각 지정하고, `copy_count`는 512Bytes에서 48MB까지 변경시켜가며 성능을 측정한 결과는 다음과 같다.

<img src="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/async-perf.png" width=800px style="display: block; margin: 0 auto; background-color: white"/>

결과를 통해 아래의 특징을 살펴볼 수 있다.
- 동기 복사(synchronous copy)는 `copy_count` 파라미터의 값이 4의 배수일 때 최상의 성능을 달성한다. 이는 컴파일러가 4개의 load 및 store instruction 그룹을 최적화할 수 있기 때문이다.
- 비동기 복사가 거의 모든 경우에 대해서 동기 복사보다 성능이 좋다.
- 비동기 복사의 경우 `copy_count` 파라미터의 값이 4의 배수일 필요가 없다.
- 전반적으로 요소의 크기가 8 or 16 바이트인 비동기 복사에서 최상의 성능을 달성한다.

> CUDA Sample [globalToShmemAsyncCopy](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/globalToShmemAsyncCopy)에서 global to shared memory async copy 성능에 대해 자세히 비교해볼 수 있다.

<br>

# Local Memory

Local memory라는 이름은 사실 물리적 위치 때문이 아닌, 스레드 내에서의 범위(scope)가 local이라고 붙여진 이름이다. 사실 local memory는 칩 외부(off-chip)에 위치하며, global memory에 액세스하는 것만큼 액세스 비용이 크다. 다시 말해, local이라는 명칭이 더 빠른 액세스를 의미하는 것은 아니다.

Local memory는 오직 automatic 변수를 저장하는 데만 사용된다. 이는 `nvcc` 컴파일러가 변수를 보관할 레지스터 공간이 충분하지 않을 때 사용한다. Local memory에 위치할 가능성이 높은 automatic 변수로는 큰 구조체이거나, 컴파일러나 동적으로 인덱싱하는 배열로 판단하는 경우이다.

PTX 어셈블리 코드(`nvcc`의 `-ptx` 또는 `-keep` 옵션으로 얻을 수 있음)를 검사하면 첫 번째 컴파일 단계에서 local memory에 배치되는 변수를 확인할 수 있다. Local memory에 배치된다면, 이는 `.local`이라는 mnemonic을 사용하여 선언되고, `ld.local`과 `st.local` mnemonics를 사용하여 액세스된다. 첫 번째 컴파일 단계에서 결정되지 않는다면, 이후의 컴파일 단계에서 변수를 보관할 레지스터 공간을 너무 많이 사용되는 것으로 판단할 때 local memory에 배치할 수도 있다. 구체적으로 어떤 변수가 지정되는지 확인할 수는 없지만, `--ptxas-options=-v` 컴파일 옵션을 통해 커널 당 사용하는 총 local memory(lmem)을 확인 할 수 있다.

> `Local memory`에 관한 내용은 [Maximize Memory Throughput: Local Memory](/cuda/doc/01_programming_guide/05-03_maximize_memory_throughput.md#local-memory)와 [CUDA Memory Model: Local Memory](/cuda/study/09_cuda_memory_model.md#local-memory)에서도 다루고 있다.

<br>

# Texture Memory

Read-only texture memory space는 캐싱된다. 따라서, 캐시 미스가 발생할 때에만 하나의 device memory read 비용이 발생한다. 캐시 미스가 발생하지 않는다면 texture cache에서 하나의 read 비용만 발생하게 된다. Texture cache는 2D spatial locality에 최적화되어 있으며, 같은 warp 내 스레드들이 가까운 texture address를 읽을 때 최적의 성능을 얻을 수 있다. 또한, 일정 지연 시간을 갖는 streaming fetch를 위해 설계되었다 (캐시 히트는 DRAM bandwidth를 줄일 뿐, fetch latency를 줄이지는 않는다).

특정 addressing 상황에서 texture fetching을 통해 device memory를 읽는 것이 global 또는 constant memory를 읽는 것보다 유리할 수 있다.

## Additional Texture Capabilities

Texture가 `tex1D()`, `tex2D()`, `tex3D()`를 사용하여 페치된다면, 하드웨어는 이미치 처리 어플리케이션에서 유용할 수 있는 몇 가지 기능들을 제공한다. 제공하는 기능은 [Table 4](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#additional-texture-capabilities__useful-features-tex1D-tex2D-tex3D-fetches)에서 확인할 수 있다. 기능을 간단히 나열하면 다음과 같다.

- **Filtering** - for fast, low-precision interpolation between texels
- **Normalized texture coordinates** - for resolution-independent coding
- **Addressing modes** - for automatic handing of boundary cases

<br>

# Constant Memory

Device에는 총 64 KB의 constant memory를 가지고 있다. 이 메모리 공간은 캐싱된다. 결과적으로 constant memory에 대한 read는 캐시 미스가 발생할 때에만 device memory로부터 하나의 memory read 비용이 발생하며, 캐시 미스가 발생하지 않는다면 상수 캐시로부터 하나의 read 비용만 발생한다. Warp 내에서 서로 다른 스레드들이 다른 constant memory 주소에 액세스하면, 직렬화, 즉, 순차적으로 처리된다. 따라서, warp 내 모든 스레드가 읽는 고유한 주소 수만큼 선형적으로 비용이 증가한다. 따라서, 상수 캐시는 같은 warp 내의 스레드들이 몇 개의 특정 주소에만 액세스하는 것이 가장 효과적이다. 만약 warp 내 모든 스레드들이 동일한 위치에 액세스한다면, 레지스터 액세스만큼 빠를 수 있다.

> `Constant memory`에 관한 내용은 [Maximize Memory Throughput: Constant Memory](/cuda/doc/01_programming_guide/05-03_maximize_memory_throughput.md#constant-memory)와 [CUDA Memory Model: Constant Memory](/cuda/study/09_cuda_memory_model.md#constant-memory)에서도 다루고 있다.

<br>

# Registers

일반적으로 레지스터에 대한 액세스는 instruction 당 추가적인 클럭 사이클을 소비하지는 않는다. 하지만, 레지스터의 read-after-write 의존성이나 register memory bank conflicts로 인한 딜레이가 발생할 수는 있다.

컴파일러와 하드웨어 스레드 스케쥴러는 register memory bank conflicts를 피하기 위해 가능한 최적으로 instruction을 스케쥴링한다. 어플리케이션에서 이러한 bank conflicts를 직접 제어할 수는 없다. 특히, 데이터를 `float4` 또는 `int4`의 벡터 데이터 타입으로 패킹하는 것은 레지스터와 관련이 없다.

## Register Pressure

주어진 작업에서 충분한 레지스터가 없을 때 **register pressure**가 발생한다. 비록 각 멀티프로세서는 수 천개의 32-bit 레지스터를 가지고 있지만, 이들은 스레드 간에 분배되는 리소스이다. 컴파일러가 너무 많은 레지스터를 할당하지 않도록 하려면, `-maxrregcount=N` 컴파일러 옵션 또는 launch bounds kernel definition qualifier를 사용하면 된다. 이를 사용하면, 스레드 당 할당되는 최대 레지스터 갯수를 지정할 수 있다.

> `Register`에 관한 내용은 [CUDA Memory Model: Registers](/cuda/study/09_cuda_memory_model.md#registers)에서도 다루고 있다.

<br>

# References

- [NVIDIA CUDA Documentation: Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)

