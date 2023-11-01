# Table of Contents

- [Table of Contents](#table-of-contents)
- [Threadblock Swizzling](#threadblock-swizzling)
- [Swizzle Logic](#swizzle-logic)
- [Profiling Result](#profiling-result)
- [References](#references)

<br>

# Threadblock Swizzling

cutlass에서는 last level cache에서 데이터 재사용을 극대화하기 위해 스레드 블록을 GEMM의 logical partition에 매핑하는데 영향을 주는 여러 함수들을 정의한다. 이 함수들에서 사용되는 기법이 바로 **swizzling** 이다. 이 기법을 통해서 L2 캐시의 hit-rate를 향상시킬 수 있다.

예를 들어, 어떤 2차원 배열이 있을때, 일반적으로 커널 구현에서 각 스레드 블록의 인덱스 배치는 아래와 같다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2020/07/default-launch-order-2d-thread-groups-1-625x192.png" height=150px style="display: block; margin: 0 auto; background-color:white"/>

위 그림에서 스레드 블록 10번과 11번을 살펴보자. 이들은 GPU에서 연속적으로 실행되며 동시에 실행될 가능성이 높다. 하지만 너비가 넓은 데이터의 경우, 스레드 블록 10에 대해서 L2에 캐시된 데이터는 스레드 블록 11에서 재사용할 수 없는 가능성이 크다.

Threadblock Swizzling의 아이디어는 위와 같은 row-major의 스레드 블록 인덱싱을 L2 캐시에 친화적인 순서로 다시 매핑하는 것이다. L2 캐시의 locality를 높이도록 스레드 블록의 ID를 swizzling하는 방법에는 여러 가지가 있으며, [Optimizing Compute Shaders for L2 Locality using Thread-Group ID Swizzling](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/)에서는 [Morton Order](https://en.wikipedia.org/wiki/Z-order_curve)를 언급하고 있고 [Locality-Aware CTA Scheduling for Gaming Applications](https://dl.acm.org/doi/fullHtml/10.1145/3477497) 논문의 **4.2 Swizzle** 에서 _return-to-start_ (Morton Order와 동일) 와 _boustrophedonic_ 이라는 두 가지 방법을 언급하고 있다.

Morton order를 통해서 스레드 블록 인덱싱을 다시 매핑하면 아래와 같이 스레드 블록 인덱스가 배치된다. 아래 그림은 한 행에서 3개의 스레드 블록이 연속적으로 배치된 이후에 다음 행에서 다시 연속적으로 3개의 스레드 블록이 배치된다.

<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2020/07/horizontal-thread-group-tiling-1-625x236.png" height=200px style="display: block; margin: 0 auto; background-color:white"/>

이전 그림과 비교했을 때, 서로 다른 행에 위치한 연속된 스레드 블록의 최대 거리가 줄어든 것을 확인할 수 있다. 이 기법을 적용하면 연속적으로 실행된 스레드 그룹 사이에서의 메모리 액세스 공간을 줄이고 GPU에서 동시에 실행되는 모든 스레드 그룹이 액세스하는 메모리가 L2 캐시에 있도록 유도할 수 있다.

# Swizzle Logic

Swizzle 인덱스를 얻는 알고리즘은 다음과 같다.

<img src="https://dl.acm.org/cms/attachment/982b83e5-1f31-488d-8de5-4d7131a04472/taco1901-01-algo1.jpg" height=600px style="display: block; margin: 0 auto; background-color:white"/>

아래의 device functiond은 위 알고리즘을 참고하여 CUDA 커널 내에서 swizzle threadblock 인덱스를 계산하는 함수로 구현한 버전이다.
```c++
__device__
uint2 get_swizzle_idx(int const tile_size)
{
    uint2 ret;

    unsigned int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    // get strip info
    unsigned int total_cta_in_a_strip = tile_size * gridDim.y;
    unsigned int number_of_strips = gridDim.x / tile_size;
    unsigned int residual_strip_width = gridDim.x % tile_size;
    // calculate swizzle CTA ID
    unsigned int strip_id = block_idx / total_cta_in_a_strip;
    unsigned int cta_id_in_strip = block_idx % total_cta_in_a_strip;
    unsigned int use_sz = (block_idx < total_cta_in_a_strip * number_of_strips) ? tile_size : residual_strip_width;
    unsigned int strip_id_x = cta_id_in_strip % use_sz;
    unsigned int strip_id_y = cta_id_in_strip / use_sz;
    unsigned int strip_flat_idx = strip_id * tile_size + strip_id_y * gridDim.x + strip_id_x;

    ret.x = strip_flat_idx % gridDim.x;
    ret.y = strip_flat_idx / gridDim.y;

    return ret;
}
```

여기서 스레드 블록은 2차원의 그리드로 커널이 실행되었다고 가정한다. 만약 `256 x 256` 크기의 2차원 배열에 대해서 `32 x 32` 크기의 스레드 블록에 대해서 각 스레드 블록을 배치한다고 가정해보자. 그리고, 하나의 행에 4개의 스레드 블록이 연속적으로 배치된 다음 다음 행에서 배치하도록 하자. 이때의 스레드 블록의 배치는 아래 그림과 같다. 왼쪽은 일반적인 스레드 블록의 인덱스 배치이고, 오른쪽은 왼쪽 인덱스를 기준으로 swizzling을 적용했을 때의 변경되는 위치를 나타낸다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcwZ4dy%2FbtszDSlHKTu%2Fd0YAOlJNXznc8T9Ja6PXG0%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

오른쪽 그림에서 4개의 스레드 블록이 연속될 때마다 행을 바꾸게 된다. 그렇게 채우다보면 2채원 배열을 4개의 열을 모두 채우게 되며, 이후에는 다음 4개의 열을 채우게 된다. swizzling하는 기준이 되는 크기(오른쪽 그림에서는 `4`)를 코드에서는 `tile_size`로 표현했다. 이 크기에 따라서 새롭게 배치된 스레드 블록은 `tile_size` 크기의 열로 구역이 나누어지게 된다. 이때 나누어지는 구역을 `strip`이라고 표현하며 오른쪽 그림에서는 2개의 `strip`이 있다. 만약 스레드 블록의 너비가 `tile_size`로 나누어 떨어지지 않는다면, 마지막 strip의 너비는 `tile_size`가 아닐 수 있다. 이 경우에는 별도의 계산이 필요하다. `get_swizzle_idx()` 함수에서는 마지막 strip에서의 너비 크기를 `residual_strip_width`로 나타내고 있으며, 마지막 strip에 포함되는 스레드 블록은 이 값을 사용하여 매핑하게 된다.

# Profiling Result

Threadblock Swizzling을 적용했을 때, L2 cache hit-rate가 얼마나 향상되는지 살펴보자. 단순 행렬 곱셈을 커널을 사용하였으며, 한 커널(`matmul_smem`)에서는 2차원 스레드 블록 인덱스를 사용하여 block tile을 인덱싱하여 계산하고, 다른 커널(`matmul_swizzle`)은 threadblock swizzling을 적용하여 새롭게 계산한 block tile 인덱스를 사용하여 계산한다.

> 사용된 코드는 [swizzle.cu](/cuda/code/threadblock_swizzle/swizzle.cu)에서 확인할 수 있다.

각 행렬 크기에 대한 결과는 아래와 같다.

- `M = N = K = 1024`

```
  void matmul_smem<(int)32>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    ------------------------------------ --------------- ------------------------------
    Memory Throughput                       Gbyte/second                          12.48
    Mem Busy                                           %                          48.10
    Max Bandwidth                                      %                          80.26
    L1/TEX Hit Rate                                    %                              0
    L2 Compression Success Rate                        %                              0
    L2 Compression Ratio                                                              0
    L2 Hit Rate                                        %                          95.93
    Mem Pipes Busy                                     %                          80.26
    ------------------------------------ --------------- ------------------------------

  void matmul_swizzle<(int)32, (int)4>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    ------------------------------------ --------------- ------------------------------
    Memory Throughput                       Gbyte/second                          14.61
    Mem Busy                                           %                          46.58
    Max Bandwidth                                      %                          77.87
    L1/TEX Hit Rate                                    %                           0.38
    L2 Compression Success Rate                        %                              0
    L2 Compression Ratio                                                              0
    L2 Hit Rate                                        %                          93.37
    Mem Pipes Busy                                     %                          77.87
    ------------------------------------ --------------- ------------------------------
```

`M = N = K = 1024`인 경우, 2차원 데이터의 너비가 작아서 기본적인 L2 cache hit-rate가 크다. `matmul_smem`의 L2 Hit Rate를 보면 이미 약 95%를 달성하고 있다. 이런 경우에 swizzle 기법을 적용해도 L2 Hit Rate 향상은 기대할 수 없으며, 결과를 보면 오히려 약 2% 가량 떨어진 것을 확인할 수 있다.

- `M = N = K = 2048`

```
  void matmul_smem<(int)32>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                         111.30
    Mem Busy                                        %                          47.46
    Max Bandwidth                                   %                          79.37
    L1/TEX Hit Rate                                 %                              0
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          51.90
    Mem Pipes Busy                                  %                          79.37
    --------------------------------- --------------- ------------------------------

  void matmul_swizzle<(int)32, (int)4>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                          34.93
    Mem Busy                                        %                          47.45
    Max Bandwidth                                   %                          79.37
    L1/TEX Hit Rate                                 %                           0.19
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          84.62
    Mem Pipes Busy                                  %                          79.37
    --------------------------------- --------------- ------------------------------
```

행렬의 크기를 조금 증가시켜 `M = N = K = 2048`인 경우에는 L2 Hit Rate의 향상을 기대할 수 있었다. 기본적인 스레드 블록 매핑을 사용한 `matmul_smem` 커널에서는 L2 Hit Rate가 51.9% 이지만, swizzle 기법을 적용한 `matmul_swizzle` 커널에서의 L2 Hit Rate는 84.62%로 32.72% 향상되었다.

다만, 두 커널의 실행 속도는 각각 9.52ms, 9.50ms로 측정되었는데, L2 캐시의 hit-rate가 속도에는 큰 영향을 미치지 않는 것으로 보인다. 이는 행렬 곱셈 알고리즘이 memory access보다 arithmetic instruction에 더 많은 영향을 받는 계산인 것으로 추정된다 (행렬 곱셈은 compute-bound 커널이다). 만약 memory-bound 커널이라면 L2 hit-rate가 높은 커널이 실행 속도가 더 빠를 것으로 기대한다.

- `M = N = K = 4096`

행렬 크기를 더 증가시켜서 테스트한 결과는 다음과 같다.

```
  void matmul_smem<(int)32>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                         118.12
    Mem Busy                                        %                          47.96
    Max Bandwidth                                   %                          80.22
    L1/TEX Hit Rate                                 %                              0
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          48.98
    Mem Pipes Busy                                  %                          80.22
    --------------------------------- --------------- ------------------------------

  void matmul_swizzle<(int)32, (int)4>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                          36.13
    Mem Busy                                        %                          47.95
    Max Bandwidth                                   %                          80.17
    L1/TEX Hit Rate                                 %                           0.09
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          84.25
    Mem Pipes Busy                                  %                          80.17
    --------------------------------- --------------- ------------------------------
```

`M = N = K = 2048`일 때와 비슷한 결과를 보여주며, 실행 속도 또한 두 커널 모두 비슷하게 측정되었다.

- `M = N = 1024`, `K = 128`

이번에는 narrow GEMM 형태의 크기로 지정하여 측정한 결과이다.

```
  void matmul_smem<(int)32>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                          29.44
    Mem Busy                                        %                          44.58
    Max Bandwidth                                   %                          74.66
    L1/TEX Hit Rate                                 %                              0
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          86.20
    Mem Pipes Busy                                  %                          74.66
    --------------------------------- --------------- ------------------------------

  void matmul_swizzle<(int)32, (int)4>(const float *, const float *, float *, int, int, int), Context 1, Stream 7
    Section: Memory Workload Analysis
    --------------------------------- --------------- ------------------------------
    Memory Throughput                    Gbyte/second                           6.58
    Mem Busy                                        %                          44.10
    Max Bandwidth                                   %                          73.77
    L1/TEX Hit Rate                                 %                           0.38
    L2 Compression Success Rate                     %                              0
    L2 Compression Ratio                                                           0
    L2 Hit Rate                                     %                          96.94
    Mem Pipes Busy                                  %                          73.77
    --------------------------------- --------------- ------------------------------
```

`M = N = K = 1024`일 때보다 `matmul_smem`의 L2 캐시의 hit-rate가 약간 감소한 것을 확인할 수 있었다. 그리고, `matmul_swizzle`에서의 L2 캐시 hit-rate는 증가했다. 일반적으로 convolution 연산을 implicit GEMM으로 구현할 때, narrow GEMM 형태가 자주 나타나는데 이러한 경우에 L2 캐시 hit-rate 향상을 기대할 수 있을 것으로 보인다.

<br>

# References

- [Optimizing Compute Shaders for L2 Locality using Thread-Group ID Swizzling](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/)
- [Locality-Aware CTA Scheduling for Gaming Applications](https://dl.acm.org/doi/fullHtml/10.1145/3477497)
- [cutlass: Efficient GEMM - Threadblock Rasterization](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#threadblock-rasterization)
- [cutlass: threadblock_swizzle.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h)