# Table of Contents

- [Table of Contents](#table-of-contents)
- [Memory Bandwidth](#memory-bandwidth)
- [Matrix Transpose Problem](#matrix-transpose-problem)
  - [Upper and Lower Performance Bound for Transpose Kernels](#upper-and-lower-performance-bound-for-transpose-kernels)
  - [Naive Transpose: read by rows vs. read by columns](#naive-transpose-read-by-rows-vs-read-by-columns)
  - [Unrolling Transpose](#unrolling-transpose)
  - [Diagonal Transpose](#diagonal-transpose)
- [Expose More Parallelism with Thin Blocks](#expose-more-parallelism-with-thin-blocks)
- [References](#references)

<br>

# Memory Bandwidth

커널 함수의 성능을 분석할 때, memory latency와 memory bandwidth를 자세히 살펴봐야 한다.

- **memory latency** : 각 memory request를 충족시키는 시간
- **memory bandwidth** : SM이 device memory 액세스할 수 있는 속도 (주로 시간당 바이트 단위로 측정됨)

커널의 성능을 향상시키기 위해서는 기본적으로 두 가지 방법을 사용할 수 있다.

첫 번째 방법은 동시에 실행되는 warp 수를 극대화하여 memory latency를 hiding하는 것이다. 이를 통해 더 많은 메모리 액세스가 계속해서 수행되도록 유지하여 bus의 saturation을 높인다.

두 번째 방법은 메모리 액세스 패턴을 정렬(aligned) 및 병합(coalesced)하여 memory bandwidth efficiency를 최대화하는 것이다. 이에 대해서는 [Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)에서 다루었다.

메모리 액세스 패턴은 직면한 문제의 특성에 따라 결정될 수 있으며 커널에 별로 좋지 않은 액세스 패턴일 수 있다. 이번 포스팅에서는 matrix transpose(행렬 전치) 문제에 대해서 어떻게 최대한 액세스 패턴을 조정하여 좋은 성능을 달성할 수 있는지에 대해서 살펴볼텐데, 그 전에 memory bandwidth에 대해서 조금 더 살펴보자.

대부분의 커널은 memory bandwidth에 민감하며, memory bandwidth에는 제한이 있다. 결과적으로 커널 성능을 튜닝할 때 memory bandwidth metrics에 집중해야 한다. Bandwidth는 global memory의 데이터가 정렬되는 방식과 warp에서 어떻게 액세스되는지에 영향을 많이 받는다.

먼저 bandwidth를 두 가지 타입으로 분류하면 아래와 같이 나눌 수 있다.

- Theoretical bandwidth
- Effective bandwidth

**Theoretical bandwidth**는 말그대로 이론적으로 측정되는 대역폭이며, 하드웨어를 통해 달성할 수 있는 최대 bandwidth를 의미한다. RTX3080의 경우, 최대 memory bandwidth는 `760GB/s`라고 한다 ([link](http://www.hwbattle.com/bbs/board.php?bo_table=hottopic&wr_id=12641) 참조).

**Effective bandwidth**는 실제로 커널이 달성할 수 있는 측정된 대역폭을 의미하며, 아래의 공식을 통해 계산할 수 있다.

$$ \text{effective bandwidth (GB/s)} = \frac{\text{(bytes read + bytes written)} \times 10^9}{\text{time elapsed}} $$

예를 들어, 4-byte 정수를 요소로 갖는 2048x2048 행렬 복사(read/write)의 경우, effective bandwidth는 아래와 같이 계산할 수 있다.

$$ \text{effective bandwidth (GB/s)} = \frac{2048 \times 2048 \times 4 \times 2 \times 10^9}{\text{time elapsed}} $$

이 공식을 사용하여 아래에서 행렬 전치 커널의 bandwidth를 측정할 예정이다.

<br>

# Matrix Transpose Problem

행렬 전치(matrix transpose)는 선형대수학에서 기본이며, 많이 사용된다. 잘 알겠지만, 행렬 전치는 아래 그림처럼 행렬의 각 행(row)을 열(column)로 바꾸는 연산이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FRVQRT%2Fbtr0Hut1Ii7%2FtzIUpuSrWuKZzBTpxNlRVK%2Fimg.png" width=300px style="display: block; margin: 0 auto"/>

Host 코드로 행렬 전치 결과를 out-place로 저장하는 가장 기본 구현은 다음과 같다.

```c++
void transposeHost(float* out, float const* in, int const nx, int const ny)
{
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx ; ix++) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}
```

기본적으로 2차원 행렬이더라도 물리적 메모리는 1차원이다. 위 함수에서는 두 개의 1차원 배열을 사용하여 배열을 저장하는데, 입력은 `in`이고 연산 결과 전치된 결과 행렬은 `out`이다. 입력 행렬의 차원은 `nx`(row) x `ny`(col) 이다. 1차원 배열에서 전치된 결과를 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeykH0Y%2Fbtr0GcmQlWs%2F6mnfqzAuX6qFfVUQTy5oa1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

위의 메모리 레이아웃에서 original matrix는 입력으로 받게 된다. 함수 내에서는 이 메모리를 read하는데, 행별로 읽게 된다. 이를 커널 함수로 만들어서 각 스레드 ID에 맞는 메모리를 읽도록 하면, 결과적으로 **coalesced access**로 메모리에 접근하게 된다.

반면 출력 행렬은 tranposed matrix는 전치할 때 열별로 메모리에 접근하여 write하게 된다. 즉, **strided access**로 메모리에 접근한다.

Strided access는 GPU 성능면에서 최악의 메모리 액세스 패턴이다. 하지만, matrix transpose 문제에서 이 액세스 패턴을 피할 수 없다. 그래도 몇 가지 방법이 있는데, 아래에서는 두 가지 버전의 행렬 전치 커널을 작성하여 bandwidth utilization을 어떻게 하면 향상시킬 수 있는지 살펴본다. 두 가지 버전의 커널 구현은 다음과 같다.

1. reads by rows and stores by columns
2. reads by columns and store by rows

아래 그림은 첫 번째 접근 방법(read by rows, store by columns)을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfZgnQ%2Fbtr0KHy8NJe%2FxhagYPsIZQ9frKRILSbJg0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

그리고 아래 그림은 두 번째 접근 방법(read by columns, store by rows)을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYGXbT%2Fbtr0JOeciz5%2FdMrCVlDYA1tOtngweCdpM0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

두 가지 방법 중 어느 방법의 성능이 더 좋을지 예측이 되는가? 

만약 L1 캐시가 비활성화되어 있다면 두 접근 방법의 성능은 이론상 동일하다. 하지만 L1 캐시가 활성화되어 있다면 두 번째 방법이 더 좋은 성능을 보여주게 된다.

두 번째 방법은 read by columns 방법으로 메모리를 읽는다. 따라서, uncoalesced read access가 발생하여 요청되지 않은 바이트로 인해 bandwidth가 낭비된다. 하지만 L1 캐시를 사용하여 global memory가 아닌 캐시에서 처리되어 속도가 빠르다. Write는 L1에 캐싱되지 않기 때문에 write by columns에서는 이러한 이점을 살릴 수 없다.

## Upper and Lower Performance Bound for Transpose Kernels

행렬 전치 커널을 구현하기 전, 먼저 두 가지 버전의 copy 커널을 구현하여 전치 커널에 대한 대략적인 성능의 상한과 하한을 계산해보자.

- Copy the matrix by loading and storing rows (upper bound)
- Copy the matrix by loading and storing columns (lower bound)

각 커널의 구현은 다음과 같다.

```c++
__global__
void coypRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__
void copyCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}
```

> 전체 코드는 [transpose.cu](/cuda/code/matrix_transpose/transpose.cu) 참조

[transpose.cu](/cuda/code/matrix_transpose/transpose.cu) 코드를 컴파일하고 컴파일하면 아래와 같은 출력 결과를 얻을 수 있다 (warmup 커널은 startup overhead를 피하기 위함).
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 2048 x 2048
warmup         elapsed time: 0.058016 ms
CopyRow        elapsed time: 0.053088 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 632.053040 GB/s
CopyCol        elapsed time: 0.066560 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 504.123077 GB/s
```

두 copy 커널의 성능은 RTX3080에서 측정되었으며, 요약하면 다음과 같다.

> 블록의 크기는 (16,16), 행렬의 크기는 (2048 x 2048)로 측정하였다.

|Kernel|Bandwidth|Ratio to Peak Bandwidth|
|--|--:|--|
|Theoretical peak bandwidth|760 GB/s||
|`CopyRow`: load/store using rows|632.1 GB/s|Upper bound: 83.2%|
|`CopyCol`: load/store using columns|504.1 GB/s|Lower bound: 66.3%|

## Naive Transpose: read by rows vs. read by columns

이번에는 아주 심플한 행렬 전치 커널을 구현한다. 먼저 아래의 커널은 load by rows, store by columns 버전의 행렬 전치 커널이다.

```c++
// 2 transpose kernel: read by rows, write by columns
__global__
void transposeNaiveRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}
```

아래 커널은 위의 `transposeNaiveRow` 커널에서 read와 write의 인덱스만 서로 바꾼 load by columns, store by rows 버전의 행렬 전치 커널이다.
```c++
// 3 transpose kernel: read by columns, write by rows
__global__
void transposeNaiveCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}
```

두 커널의 실행 결과는 다음과 같다.
> 블록의 크기는 (16,16), 행렬의 크기는 (2048 x 2048)로 측정
```
NaiveRow       elapsed time: 0.073728 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 455.111084 GB/s
NaiveCol       elapsed time: 0.066272 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 506.313873 GB/s
```

|Kernel|Bandwidth|Ratio to Peak Bandwidth|
|--|--:|:--:|
|Theoretical peak bandwidth|760 GB/s||
|`NaiveRow`: load rows/store columns|455.1 GB/s|59.9%|
|`NaiveCol`: load columns/store rows|520.6 GB/s|68.5%|

위의 결과를 통해 `NaiveCol` 커널의 성능이 `NaiveRow` 커널의 성능보다 더 좋다는 것을 볼 수 있다. 앞서 언급했듯이, read 메모리 연산은 L1 캐시에 캐싱되지만, write 메모리 연산은 캐싱되지 않는다. 따라서, `NaiveCol`에서 read 연산은 strided access 패턴이지만 L1 캐시를 통해 빠른 read가 가능하다. 그리고 write 연산은 L1 캐시에 캐싱되지는 않지만 coalesced access 패턴으로 연산되므로 최적의 메모리 액세스 패턴을 만족한다.

반면, `NaiveRow` 커널에서는 read 연산이 coalesced access 패턴이고, write 연산은 strided access 패턴이다. Read 연산은 병합된 메모리 액세스로 빠를 수 있지만, write 연산은 strided access 패턴으로 GPU에서 최악의 메모리 패턴으로 액세스하게 된다. 그 결과, `NaiveCol`보다 성능이 더 좋지 않다.

이러한 차이는 결과적으로 L1 캐시에 의한 것이다. 컴파일 할 때, L1 캐시를 비활성화하면 두 커널의 성능은 엇비슷하게 측정되는 것을 확인할 수 있다.

> L1 캐시를 비활성화하려면, `-Xptxas -dlcm=cg` 옵션을 추가해 컴파일하면 된다. L1 캐시를 비활성화하고 두 커널 성능의 측정 결과는 다음과 같다.
> ```
> NaiveRow       elapsed time: 0.074752 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 448.876709 GB/s
> NaiveCol       elapsed time: 0.075680 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 443.372498 GB/s
> ```

`nsight compute`를 통해 지금까지 구현한 커널에 대해 load/store throughput을 측정해보자.

```
$ ncu --metrics=l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second ./transpose

copyRow(float *, const float *, int, int), Context 1, Stream 
  Section: Command line profiler metrics
  ------------------------------------------------------------ --------------- ----------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second         Gbyte/second     344.02
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second         Gbyte/second     344.02
  ------------------------------------------------------------ --------------- ----------
copyCol(float *, const float *, int, int), Context 1, Stream 
  Section: Command line profiler metrics
  ------------------------------------------------------------ --------------- ----------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second         Gbyte/second     982.73
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second         Gbyte/second     982.73
  ------------------------------------------------------------ --------------- ----------
transposeNaiveRow(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ------------------------------------------------------------ --------------- ----------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second         Gbyte/second     202.98
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second         Gbyte/second     811.91
  ------------------------------------------------------------ --------------- ----------
transposeNaiveCol(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ------------------------------------------------------------ --------------- ----------
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second         Tbyte/second       1.12
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second         Gbyte/second     280.37
  ------------------------------------------------------------ --------------- ----------
```

|Kernel|Load Throughput|Store Throughput|Note|
|--|--:|--:|--|
|`CopyRow`: load/store using rows|344.02 GB/s|344.02 GB/s|Upper bound|
|`CopyCol`: load/store using columns|982.73 GB/s|982.73 GB/s|Lower bound|
|`NaiveRow`: load rows/store columns|202.98 GB/s|811.91 GB/s|Strided write / Coalesced read|
|`NaiveCol`: load columns/store rows|1120 GB/s|280.37 GB/s|Strided read / Coalesced write|

위 결과를 통해 캐싱된 strided read에서 가장 높은 처리량을 얻을 수 있다는 것을 보여준다. Cached read는 L1 캐시에 128바이트의 캐시라인으로 처리된다. Read by columns라면 각 warp에서의 memory transaction request는 32번이 발생된다. 결과적으로 in-flight global memory read로부터 hiding latency 효과가 커지고, 한 번 L1으로 pre-fetch되면 높은 L1 cache-hit를 보여주기 때문에 성능이 좋다.

다음으로 `nsight compute`를 통해 load/store 효율을 측정해보자.
```
$ ncu --metrics=smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./transpose

copyRow(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------------- ---- -----
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct      %   100
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct      %   100
  --------------------------------------------------------------- ---- -----
copyCol(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------------- ---- -----
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct      %    25
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct      %    25
  --------------------------------------------------------------- ---- -----
transposeNaiveRow(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------------- ---- -----
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct      %   100
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct      %    25
  --------------------------------------------------------------- ---- -----
transposeNaiveCol(float *, const float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------------- ---- -----
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct      %    25
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct      %   100
  --------------------------------------------------------------- ---- -----
```

|Kernel|Load Efficiency|Store Efficiency|
|--|--:|--:|
|`CopyRow`: load/store using rows|100 %|100 %|
|`CopyCol`: load/store using columns|25 %|25 %|
|`NaiveRow`: load rows/store columns|100 %|25 %|
|`NaiveCol`: load columns/store rows|25 %|100 %|

위 결과를 통해 `NaiveCol` 구현에서 coalesced write를 통해 낭비되는 bandwidth가 없지만, strided read에서는 낭비되는 bandwidth가 있다는 것을 보여준다. 하지만, load efficiency가 좋지 않더라도 L1 캐시에 캐싱되면 strided read로 인한 성능 하락을 막을 수 있다.

## Unrolling Transpose

이번에는 unrolling 기법을 통해 memory bandwidth utilization을 향상시켜보자. Unrolling의 목적은 각 스레드가 더 많은 독립적인 작업들을 수행하도록 하여 in-flight memory request를 최대화하는 것이다.

> Unrolling 기법에 대해서는 [Unrolling Loops](/cuda/study/08_unrolling_loops.md)에서 다루고 있음

아래 코드는 unrolling factor를 4로 설정하여 구현한 커널이다. row-based, column-based으로 각각 구현하였다.

```c++
// 4 transpose kernel: read by rows, write by columns + unroll 4 blocks
__global__
void transposeUnroll4Row(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in [ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// 5 transpose kernel: read by columns, write by rows + unroll 4 blocks
__global__
void transposeUnroll4Col(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[ti] = in [to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}
```

두 커널은 하나의 스레드 블록이 여러 개의 데이터 블록을 처리하므로, 그리드의 x 차원 크기를 다시 계산해주어야 한다 (1/4 크기로).

위 커널의 성능은 다음과 같이 측정되었다.
```
Unroll4Row     elapsed time: 0.078080 ms <<< grid(32,128) block(16,16)>>> effective bandwidth: 429.744263 GB/s
Unroll4Col     elapsed time: 0.059520 ms <<< grid(32,128) block(16,16)>>> effective bandwidth: 563.750549 GB/s
```

|Kernel|Bandwidth|Ratio to Peak Bandwidth|
|--|--:|:--:|
|Theoretical peak bandwidth|760 GB/s||
|`NaiveRow`: load rows/store columns|455.1 GB/s|59.9%|
|`NaiveCol`: load columns/store rows|520.6 GB/s|68.5%|
|`Unroll4Row`: load rows/store columns|429.7 GB/s|56.5%|
|`Unroll4Col`: load columns/store rows|563.7 GB/s|74.1%|

L1 캐시로 인해 `Unroll4Col` 구현의 성능이 더 좋으며 기존 구현인 `NaiveCol`보다 더 좋은 성능을 보여준다. `Unroll4Row`의 경우에는 `NaiveRow` 구현의 성능과 큰 차이가 없으며, 오히려 더 느리게 측정되었다.

## Diagonal Transpose

커널을 실행(launch)할 때, 스레드 블록들은 SM들간에 분배된다. 프로그래밍 모델 측면에서 그리드는 1차원 또는 2차원 레이아웃으로 나타낼 수 있지만, 하드웨어 관점에서 모든 블록들은 1차원으로 정렬된다. 각 블록은 저마다의 유일한 식별자(`bid`)가 있으며, 이는 다음과 같이 row-major ordering으로 계산할 수 있다.

```c++
int bid = blockDim.x * blockIdx.y + blockIdx.x;
```

아래 그림은 4x4 그리드에서 각 스레드 블록에 대응되는 block ID를 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbxbcl4%2Fbtr0RJJSXw9%2FVzphS3Y5L0OFdZbGbO9Qqk%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

커널이 실행되면, SM에 할당되는 스레드 블록의 순서는 block ID를 통해 결정된다. 모든 SM이 full occupancy 상태라면, 남은 스레드 블록들은 진행 중인 스레드 블록이 완료될 때까지 홀딩되며, SM이 수행 중인 스레드 블록이 완료되면 다른 스레드 블록이 이 SM에 할당된다. 스레드 블록이 완료되는 속도와 순서는 정해져있지 않기 때문에 초기에 active thread block들은 연속적일 수 있지만 시간이 지날수록 연속적이지 않게 된다.

스레드 블록이 스케줄링되는 순서를 직접 제어할 수는 없지만, 블록 좌표인 `blockIdx.x`와 `blockIdx.y`는 유연하게 해석할 수 있다. 위 그림은 데카트르 좌표(cartesian coordinate)를 사용하여 블록 좌표를 나타냈는데, 아래 그림은 diagonal coordinate(대각 좌표)으로 블록 좌표를 나타내는 것을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FywBq5%2Fbtr0NJpQz2Y%2FzlzuufTw08YryWLzkippdK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

대각 좌표계를 통해 1D thread block ID를 결정하지만, 데이터 액세스를 위해서는 여전히 데카르트 좌표계를 사용해야 한다. 따라서, block ID는 대각 좌표로 해석하되, 데이터 블록의 데이터를 액세스할 때는 대각 좌표를 데카르트 좌표로 매핑해야 한다. 정사각행렬에서 대각->데카르트 좌표 매핑을 통해 block ID는 아래와 같이 계산할 수 있다.

```c++
unsigned int block_y = blockIdx.y;
unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
```

이렇게 데카르트 좌표로 계산된 block ID를 통해 처리해야할 데이터의 인덱스를 기존과 같은 방식으로 계산할 수 있다.

대각 좌표를 사용하여 구현한 행렬 전치 커널은 다음과 같다. 각각 row-based, column-based로 구현된 커널이다.

```c++
// 6 transpose kernel: read by rows, write by columns + diagonal coordinate transform
__global__
void transposeDiagonalRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int block_y = blockIdx.y;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.y;

    unsigned int ix = blockDim.x * block_x + threadIdx.x;
    unsigned int iy = blockDim.y * block_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// 7 transpose kernel: read by columns, write by rows + diagonal coordinate transform
__global__
void transposeDiagonalCol(float* out, float const* in, int const nx, int const ny)
{
    unsigned int block_y = blockIdx.y;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.y;

    unsigned int ix = blockDim.x * block_x + threadIdx.x;
    unsigned int iy = blockDim.y * block_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}
```

두 커널의 성능은 아래와 같이 측정된다.

```
DiagonalRow    elapsed time: 0.076032 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 441.319885 GB/s
DiagonalCol    elapsed time: 0.065728 ms <<< grid(128,128) block(16,16)>>> effective bandwidth: 510.504364 GB/s
```

|Kernel|Bandwidth|Ratio to Peak Bandwidth|
|--|--:|:--:|
|Theoretical peak bandwidth|760 GB/s||
|`NaiveRow`: load rows/store columns|455.1 GB/s|59.9%|
|`NaiveCol`: load columns/store rows|520.6 GB/s|68.5%|
|`Unroll4Row`: load rows/store columns|429.7 GB/s|56.5%|
|`Unroll4Col`: load columns/store rows|563.7 GB/s|74.1%|
|`DiagonalRow`: load rows/store columns|441.3 GB/s|58.1%|
|`DiagonalCol`: load columns/store rows|510.5 GB/s|67.2%|

`DiagonalCol` 커널을 통해 스레드 블록의 실행 순서를 변경해도 column 기반의 커널에서 달성한 것과 유사한 성능을 얻을 수 있다. 다만, 구현은 데카르트 좌표를 쓰는 것보다 복잡할 수 있다.

# Expose More Parallelism with Thin Blocks

더 많은 병렬 처리를 수행하도록 하는 가장 간단한 방법은 블록의 크기를 조정하는 것이다. 지금까지는 (16,16) 크기에 대해서 측정했고, column-based 커널의 성능이 가장 좋다는 것을 살펴봤다. [transpose.cu](/cuda/code/matrix_transpose/transpose.cu)를 컴파일하고, 실행할 때 처음 두 개의 인자를 통해 블록의 크기를 지정할 수 있다. (8,8)부터 (32,32)까지의 블록 크기에 대한 `NaiveCol` 커널의 성능은 다음과 같다.

> 행렬 크기는 (2048 x 2048)

|Kernel|Block Size|Bandwidth|
|--|:--:|--:|
|`NaiveCol`|(8,8)|430.45 GB/s|
|`NaiveCol`|(8,16)|538.84 GB/s|
|`NaiveCol`|(8,32)|577.09 GB/s|
|`NaiveCol`|(16,8)|474.89 GB/s|
|`NaiveCol`|(16,16)|514.51 GB/s|
|`NaiveCol`|(16,32)|532.27 GB/s|
|`NaiveCol`|(32,8)|448.49 GB/s|
|`NaiveCol`|(32,16)|450.22 GB/s|
|`NaiveCol`|(32,32)|364.34 GB/s|

위 결과에 의하면 최적의 블록 크기는 (8,32)이다. 동일한 크기은 (16,16)보다 더 좋은 성능을 보여준다. 이와 같이 'thin' block dim에서 더 좋은 성능을 보여주는 이유는 아래 그림을 통해 설명할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FctExuE%2Fbtr0JBFUmXB%2FDmshkUT5ZkMLYi3llQTTl0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

'thin' 블록은 스레드 블록의 innermost 차원의 값이 크기 때문에 스레드 블록에 의해서 저장되는 연속적인 요소의 수가 증가되어 store operation의 효율성을 향상시킨다. 블록의 크기를 (8,32)로 지정하여 커널을 실행한 결과는 다음과 같다.
```
$ ./transpose 8 32

> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 2048 x 2048
warmup         elapsed time: 0.062144 ms
CopyRow        elapsed time: 0.053248 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 630.153870 GB/s
CopyCol        elapsed time: 0.055232 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 607.517944 GB/s
NaiveRow       elapsed time: 0.057344 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 585.142822 GB/s
NaiveCol       elapsed time: 0.057632 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 582.218750 GB/s
Unroll4Row     elapsed time: 0.060416 ms <<< grid(64,64) block(8,32)>>> effective bandwidth: 555.389832 GB/s
Unroll4Col     elapsed time: 0.057376 ms <<< grid(64,64) block(8,32)>>> effective bandwidth: 584.816528 GB/s
DiagonalRow    elapsed time: 0.039424 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 851.116943 GB/s
DiagonalCol    elapsed time: 0.036704 ms <<< grid(256,64) block(8,32)>>> effective bandwidth: 914.190063 GB/s
```

|Kernel|Bandwidth|Ratio to Peak Bandwidth|
|--|--:|:--:|
|Theoretical peak bandwidth|760 GB/s||
|`CopyRow`: load/store using rows|630.2 GB/s|82.9%|
|`CopyCol`: load/store using columns|607.5 GB/s|79.8%|
|`NaiveRow`: load rows/store columns|585.1 GB/s|76.9%|
|`NaiveCol`: load columns/store rows|582.2 GB/s|76.6%|
|`Unroll4Row`: load rows/store columns|555.4 GB/s|73.1%|
|`Unroll4Col`: load columns/store rows|584.8 GB/s|76.9%|
|`DiagonalRow`: load rows/store columns|851.1 GB/s|111.9%|
|`DiagonalCol`: load columns/store rows|914.2 GB/s|120.3%|

> 블록 크기 (8, 32), 행렬 크기 (2048 x 2048)

블록 크기를 (8,32)로 지정한 결과는 꽤나 인상적이다. 기존에는 row-based가 column-based 구현보다 성능이 좋지 않았는데, 블록의 크기를 (8,32)로 바꿨을 때 더 좋아지는 경우가 발생한다. 또한, 대각 좌표를 사용하여 스레드 블록의 실행 순서에 변경을 준 `DiagonalRow`와 `DiagonalCol` 커널의 성능이 대폭 향상된 것을 볼 수 있다.

> `DiagonalRow`와 `DiagonalCol` 커널의 성능이 (8,32) 블록 크기에서 대폭 향상된 정확한 이유는 현재로서는 잘 모르겠다. 다만, load/store bandwidth를 측정했을 때, load와 store의 bandwidth가 모두 높은 경향을 보인다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher