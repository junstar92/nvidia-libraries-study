# Table of Contents

- [Table of Contents](#table-of-contents)
- [Memory Access Patterns](#memory-access-patterns)
- [Aligned and Coalesced Access](#aligned-and-coalesced-access)
- [Global Memory Reads](#global-memory-reads)
  - [Cached Loads](#cached-loads)
  - [Cached Loads without L1 cache](#cached-loads-without-l1-cache)
  - [Example: Misaligned Reads](#example-misaligned-reads)
- [Global Memory Writes](#global-memory-writes)
  - [Example: Misaligned Writes](#example-misaligned-writes)
- [Array of Structures vs Structure of Arrays](#array-of-structures-vs-structure-of-arrays)
  - [Example: AoS Data Layout](#example-aos-data-layout)
  - [Example: SoA Data Layout](#example-soa-data-layout)
- [References](#references)

<br>

# Memory Access Patterns

대부분의 device data access는 global memory로부터 시작되며, global memory의 bandwidth는 제한되어 있다. 따라서, 커널의 성능을 끌어올리는 첫 번째 단계는 global memory bandwidth를 최대한 활용하는 것이다. Global memory 사용을 적절하게 조정하지 않는다면, 다른 최적화를 적용해도 그 영향이 미미할 수 있다.

Data를 읽고 쓰는데 최적의 성능을 달성하기 위해서 memory access operation은 반드시 특정 조건들을 만족해야 한다. [CUDA execution model](/cuda/study/05_cuda_execution_model.md)의 특징 중 하나는 warp 단위로 instruction이 실행(issued/executed)된다는 것이다. Memory operation 또한 warp 단위로 실행된다. 

> instruction issue에 대한 정확한 의미는 stackoverflow([link](https://stackoverflow.com/a/49923841))에서 자세히 살명하고 있다.

Memory instruction이 실행될 때, warp 내 각 스레드들은 읽거나 저장할 메모리 주소를 제공한다. Warp 내 32개의 스레드들은 서로 협력하여 요청된 메모리 주소들로 구성된 하나의 memory access request를 보내며, 이는 하나 이상의 device memory transaction을 통해 처리된다. 즉, 32개의 메모리 주소에 액세스해야 하지만 한 번의 memory transaction으로 처리할 수도 있다는 것을 의미한다.

Warp 내에서 메모리 주소 분포에 따라 메모리 액세스는 여러 패턴으로 분류된다. 이번 포스팅에서는 어떠한 종류의 메모리 액세스 패턴들이 있는지 살펴보고, 최적의 성능을 위해서는 어떤 액세스 패턴을 사용해야 되는지 알아본다.

<br>

# Aligned and Coalesced Access

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FsxJ2h%2FbtrZK1GjW4d%2FaKkrGPIhuROP1yxaKOYHZ1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림을 통해서 알 수 있듯이 global memory access는 항상 L2 cache에 캐싱되며, global memory는 모든 커널에서 액세스할 수 있는 logical memory space이다. 모든 데이터는 처음에 DRAM이라는 physical device memory에 상주하게 된다. 커널의 memory request는 일반적으로 device DRAM과 SM on-chip memory 간에 처리되는 것이고, 32-, or 128-bytes memory transaction으로 액세스된다.

> [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)에 의하면 64-byte memory transaction으로도 액세스된다고 언급하고 있음

Compute capability가 6.0 이상인 device인 경우, 한 warp 내 스레드들의 concurrent access는 warp 내 모든 스레드들에게 필요한 32-bytes transaction의 수와 동일한 수의 transaction으로 통합된다.

Compute capability가 5.2인 특정 device들은 L1-caching이 선택적으로 활성화될 수 있다. 이러한 device에서 L1-caching이 활성화되면 스레드들에 의해 요청되는 transaction의 수는 필요한 128-byte aligned segment의 수와 동일하다.

> Compute capability 6.0 이상에서는 L1-caching이 default이다. 하지만, **L1에 캐싱되는지 여부에 상관없이 data access unit은 32-byte**이다. 이어지는 내용부터는 compute capability 6.0 이상의 경우에 대해서만 살펴본다. 5.2는 요즘에는 거의 사용되지 않는다.

L1 캐시 라인은 128 bytes이며, 이는 device memory의 128-byte aligned segment에 매핑된다. 만약 warp 내 각 스레드들이 하나의 4-byte 값을 요청한다면, 각 요청마다 128 bytes라는 것이며 이는 완벽히 캐시 라인과 device memory segment의 크기와 일치한다. 참고로 compute capability 6.0 이상인 device에서 data access unit이 32 bytes 이므로, 이 경우에는 4개의 transaction이 요청된다는 것을 알 수 있다.

CUDA 성능 최적화를 위해서 알고 있어야 하는 device memory 액세스의 두 가지 특징은 다음과 같다.

- Aligned memory accesses
- Coalesced memory accesses

**Aligned memory accesses**는 device memory transaction의 첫 번째 주소가 ~~캐시 단위의 배수일 때 발생한다. L2 캐시의 경우 32 bytes이고, L1 캐시인 경우에는 128 bytes이다~~ memory transaction 크기(**32 bytes**)의 배수일 때 발생한다. 만약 misaligned load가 발생하면 bandwidth가 낭비된다.

**Coalesced memory accesses**는 직역하면 병합된 메모리 액세스라고 해석할 수 있는데, 이러한 메모리 액세스는 한 warp 내 모든 스레드들이 연속된 메모리 덩어리에 액세스할 때 발생한다.

**Aligned coalesced memory accesses**가 가장 이상적이다. Global memory throughput을 최대화하려면 memory operation이 정렬(aligned)되면서 병합(coalesced)되도록 해야 한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FVJLNv%2FbtrZLTg0Ueb%2FLpA02bk4HoTE8Q89yZiex1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림은 정렬(aligned)되면서 병합(coalesced)된 memory load operation을 보여준다. 이 경우, device memory로부터 데이터를 읽기 위해 4개의 32-byte transaction이 요청된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fea4wpZ%2FbtrZJVGqprf%2F2ZkSMrc1udjLJTl1cnrtI0%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림은 정렬되지 않고, 병합되지도 않은 메모리 액세스를 보여준다. 이 경우에는 정렬, 병합된 액세스보다 더 많은 transaction이 발생할 것이다.

성능을 최대화하려면 memory transaction 효율을 최적화해야 한다. 즉, 최소한의 memory transaction으로 최대한의 memory request를 처리해야 한다.

# Global Memory Reads

SM(streaming multiprocessor)에서 데이터는 메모리 타입에 따라 아래의 캐시/버퍼를 통해 파이프라인된다.

- L1/L2 cache
- Constant cache
- Read-only(Texture) cache

기본적으로 L1/L2 캐시가 사용되는데, 이는 컴파일 옵션이나 device에 따라 약간씩 다를 수 있다. Global memory access는 항상 L2에 캐시되며, 컴파일 옵션에 따라 L1 캐시가 활성화 또는 비활성화될 수 있다 (기본적으로는 활성화되어 있음). L1 캐시 비활성화에 대해서는 다루지 않으며, L1 캐시는 기본적으로 사용되도록 한다.

## Cached Loads

L1 캐시를 통한 load operation은 128 바이트의 L1 cache line 단위의 device memory transaction으로 처리된다. Cached load 또한 aligned/misaligned, coalesced/uncoalesced로 분류될 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbzri2J%2FbtrZLVyYoVB%2FqckYZO1KPd4d2amQE2kc7K%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림은 가장 이상적인 정렬(aligned) + 병합(coalesced)된 메모리 액세스이다. Warp 내 모든 스레드들에 의해 요청되는 주소가 128바이트의 캐시 라인에 딱 맞아 떨어지는 경우이다. 앞서 언급했듯이 데이터 액세스 단위는 32바이트이므로, 4개의 32-byte memory transaction으로 이 메모리 요청이 처리된다. 4개의 transaction에서 사용되지 않는 데이터가 없으므로 global memory load 효율은 100%이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fwdekj%2FbtrZ2z80FDk%2FWGtzI8mIxV4k2wzdzKxsA0%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위의 경우는 메모리 주소가 정렬은 되었지만 warp 내의 각 스레드들이 ID에 연속적이지 않고 랜덤한 주소를 참조하는 경우이다. 각 스레드들에서 요청되는 주소는 여전히 캐시 라인에 딱 맞아떨어지기 때문에 4개의 memory transaction으로 모두 처리된다. 각 스레드들이 랜덤한 주소지만 별도의 4바이트 데이터들을 처리하는 한 global memory load 효율은 여전히 100%이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbRoDVg%2FbtrZ0JcQ2TN%2F8DptKkkpGm64oUxdUh3lL1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림은 warp가 정렬되지 않은 연속된 32개의 4바이트 데이터를 요청하는 경우이다. 캐시 라인 크기에 딱 맞아 떨어지지 않고, 2개의 128바이트 세그먼트에 필요한 메모리가 나누어져 있다. SM에서 물리적인 load operation은 반드시 128바이트로 정렬되어야 하므로 2개의 128바이트 세그먼트로 읽는다. 그리고 memory transaction은 32바이트 단위이므로 offset이 32로 나누어 떨어지지 않는다면 4번의 memory transaction이 5번으로 증가하게 된다. 따라서, global memory load 효율은 80%가 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fr3BSN%2Fbtr0o6sFyc7%2Fl2yN3qW9EZJwaoMvwMPWqK%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위 그림은 warp 내 모든 스레드들이 하나의 주소만 참조하는 경우이다. 캐시 라인에 딱 맞아 떨어질 것이고, 1번의 memory transaction만으로 충분하다. 단 사용되는 데이터는 4바이트뿐이지만, memory transaction은 32바이트 단위이다. 따라서 global memory load 효율은 12.5%(4/32)가 된다.

## Cached Loads without L1 cache

L1 캐시가 사용되지 않는 경우에는 L2 캐시를 통해 데이터를 로드한다. 알려진 바로 L2 cache line의 크기는 32-bytes라고 한다. 최근 GPU (compute capability 6.0 이상)에서 L1, L2와 상관없이 memory transaction은 32바이트 단위이므로 L2를 사용하더라도 각 패턴에 대해 global memory load 효율은 L1을 사용하는 것과 동일하다고 볼 수 있다. 단, 캐시에 따른 효율은 당연히 달라진다.

## Example: Misaligned Reads

액세스 패턴에 따라서 어떤 영향을 미치는지 살펴본다. 아래의 `readOffset` 커널을 사용하는데, 이 커널은 offset을 파라미터로 전달받아 offset만큼 떨어진 데이터 하나를 읽고 다른 곳에 저장한다.

```c++
__global__
void readOffset(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        b[i] = a[k];
}
```

> 전체 코드는 [read_segment.cu](/cuda/code/global_access_test/read_segment.cu)를 참조

데이터 배열 요소의 수는 128(512 bytes)개로 지정했고, `readOffset` 커널은 `<<<1,32>>>`(1 grid, 32 thread blocks)로 호출된다. 커널은 항상 32개의 스레드(1 warp)로만 호출되고 `float` 타입의 데이터 하나를 읽고 저장하므로, 이 커널이 수행되면 128바이트의 데이터를 요청한다.

테스트를 위해 `nsight compute`를 통해 커널 함수를 프로파일링하는데, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`와 `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` 메트릭을 측정한다.

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` : memory transaction 수 측정
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` : global memory load operation 효율 측정

먼저 offset을 0으로 지정하여 프로파일링한 결과이다. Offset이 0이므로 aligned and colaseced memory access를 수행한다.

```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./read_segment 0

readOffset(float *, float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                      sector       4
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct            %     100
  ---------------------------------------------------------------- --------- -------
```

이 커널에서 요청된 128바이트에 대해 4개의 memory transaction이 발생했다. Memory transaction 단위는 32바이트이므로 4번의 memory transaction이 발생했고, 모든 데이터가 사용되었으므로 효율은 100%로 측정된 것이다.

다음은 offset을 1로 지정한 결과이다.

```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./read_segment 1

readOffset(float *, float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                      sector       5
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct            %      80
  ---------------------------------------------------------------- --------- -------
```

Offset으로 인해 misaligned이 발생하게 된다. 시작 주소가 이제는 memory transaction 단위인 32 bytes의 배수로 시작되지 않는다. 따라서, 32바이트 단위로 구분했을 때, 5개의 32바이트 영역을 읽어야만 요청한 데이터를 모두 읽을 수 있다. 따라서, memory transaction의 수는 5번으로 측정된다. 하지만, 필요한 데이터는 사실 4번의 memory transaction으로 처리 가능한 크기이므로, 효율은 `4/5=0.8(80%)`로 측정된다.

그렇다면 실제로 misaligned access를 피하려면 32바이트의 배수로 시작하는 주소를 전달하면 되는지 확인해보자. 이번에는 offset 크기를 8로 지정한다. 배열 요소 하나의 크기는 4바이트이므로, 8로 지정하면 32바이트의 offset이 발생하게 된다. 그 결과는 다음과 같다.
```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./read_segment 8

readOffset(float *, float *, int, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                      sector       4
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct            %     100
  ---------------------------------------------------------------- --------- -------
```

실제로 측정해보면, offset이 0일 때와 결과가 같다는 것을 확인할 수 있다.

> 간단한 vector addition 커널을 작성하여 offset 테스트했을 때, misaligned memory access의 런타임 실행 시간은 aligned memory access와 큰 차이가 없었다. 
>
> 문서에 따르면 misaligned access가 발생했을 때의 memory throughput은 aligned access의 약 4/5 정도라고 한다. 하지만 특정 예제에서 offset memory throughput은 약 9/10 정도의 성능 하락만 발생하는데, 인접한 warp가 근처에 fetch된 cache line을 재사용하기 때문이다. 따라서, misaligned access의 영향이 있긴 하지만 높은 수준의 cache line reuse로 인해 예상한 것만큼의 성능 하락은 없다고 해석할 수 있을 것 같다.

<br>

# Global Memory Writes

Memory store operation은 비교적 간단하다. L1 캐시는 store operation에서 사용되지 않으며 오직 L2 캐시에만 캐싱된다. Memory transaction 단위는 여전히 32 bytes로 동일하다. 따라서, 메모리 액세스 패턴은 [Global Memory Reads](#global-memory-reads)에서 설명한 것과 동일하다.

## Example: Misaligned Writes

아래의 `writeOffset` 커널을 통해 memory write operation에서의 misaligned access에 따른 성능 비교를 수행한다.

```c++
__global__
void writeOffset(float* a, float* b, int const n, int const offset)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n)
        b[k] = a[i];
}
```

> 전체 코드는 [write_segment.cu](/cuda/code/misaligned_access_test/write_segment.cu)를 참조

[Example: Misaligned Reads](#example-misaligned-reads)와 동일한 조건으로 테스트를 수행했다. 이번에는 `nsight compute`를 통해 프로파일링할 때, `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`와 `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` 메트릭을 사용한다. 이 메트릭은 각각 memory store transaction의 수와 global memory store operation 효율을 나타낸다.

Offset을 0으로 지정했을 때의 결과는 다음과 같다.
```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./write_segment 0
writeOffset(float *, float *, int, int),  Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                      sector       4
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct            %     100
  ---------------------------------------------------------------- --------- -------
```

메모리 주소가 정렬되어 있기 때문에 128바이트 메모리 주소에 대한 요청은 4번의 memory transaction으로 모두 처리할 수 있다. 그리고 memory transaction에서 모든 데이터가 사용되었기 때문에 효율은 100%이다.

이번에는 offset을 1로 지정했을 때의 결과이다.
```
ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./write_segment 1
writeOffset(float *, float *, int, int),  Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                      sector       5
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct            %      80
  ---------------------------------------------------------------- --------- -------
```
Offset으로 인해 misalinged memory access가 발생한다. 따라서 4번이 아닌 5번의 memory transaction이 필요하게 되고, 그 결과, 효율은 80%가 된다.

마찬가지로 offset을 8로 지정하여 32바이트의 메모리 offset을 지정해주면, offset이 0일 때와 동일한 결과를 얻을 수 있다.
```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./write_segment 8
writeOffset(float *, float *, int, int),  Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- --------- -------
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                      sector       4
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct            %     100
  ---------------------------------------------------------------- --------- -------
```

<br>

# Array of Structures vs Structure of Arrays

데이터를 구조화하는 방법 중 **array of structure(AoS)**와 **structure of arrays(SoA)** 패턴이 있다. 각 패턴은 구조화된 데이터를 저장할 때 각각의 장점이 있따.

간단하게 한 쌍의 데이터 요소를 저장하는 구조체를 정의한다고 할 때, 먼저 AoS 패턴은 다음의 `innerStruct`처럼 정의하는 것이다.

```c++
struct innerStruct {
    float x;
    float y;
};
struct innerStruct myAoS[N];
```

SoA 패턴 방식으로 구현하면 아래의 `innerArray`와 같이 정의할 수 있다.
```c++
struct innerArray {
    float x[N];
    float y[N];
};
struct innerArray mySoA;
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd3l9Fz%2FbtrZNuH4R6X%2FKP9KmiCnbhaSuT008PZ3J1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

위 그림은 AoS와 SoA 패턴의 메모리 레이아웃을 보여준다. GPU에서 AoS 포맷을 사용하여 데이터 구조체의 x 멤버에만 액세스한다고 가정해보면 32-bytes memory transaction에 y 멤버가 포함되기 때문에 50%의 bandwidth 낭비가 발생하게 된다.

반면, 동일한 조건에서 SoA 포맷을 사용하면 GPU memory bandwidth를 낭비없이 사용할 수 있다. x 멤버가 떨어져 있지 않고, 연속된 메모리 상에 존재하기 때문에 coalesced memory access를 달성할 수 있다. 따라서, 더 효율적인 global memory utilization이 가능하다.

## Example: AoS Data Layout

아래의 간단한 커널을 구현해서 AoS 레이아웃의 성능을 살펴보자. 구조체 배열은 global memory에 x, y가 번갈아가며 선형 메모리로 저장된다.

```c++
__global__
void testInnerStruct(innerStruct* data, innerStruct* result, int const n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}
```

> 전체 코드는 [test_AoS.cu](/cuda/code/misaligned_access_test/test_AoS.cu)를 참조

위 커널 코드를 컴파일하고 `nsight compute`를 통해 global memory load/store 효율을 측정하면 아래와 같은 결과를 얻을 수 있다.
```
$ ncu --metrics=smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./test_AoS

testInnerStruct(innerStruct *, innerStruct *, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- ------ ------
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct         %     50
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct         %     50
  ---------------------------------------------------------------- ------ ------
```

측정 결과, 예상한대로 global memory load와 store의 효율이 각각 50%로 측정되는 것을 볼 수 있다. 이는 array of structure 구조체의 각 멤버에 접근할 때, 필요하지 않은 다른 멤버까지 memory transaction에 포함되므로 bandwidth 낭비에 의한 결과이다.

## Example: SoA Data Layout

[Example: AoS Data Layout](#example-aos-data-layout)와 동일한 조건으로 아래의 커널을 구현하여 테스트한다.

> 전체 코드는 [test_SoA.cu](/cuda/code/misaligned_access_test/test_SoA.cu)를 참조

```c++
__global__
void testInnerArray(innerArray* data, innerArray* result, int const n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}
```

커널의 global memory load/store efficiency를 측정한 결과는 다음과 같다.

```
$ ncu --metrics=smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./test_SoA

testInnerArray(innerArray *, innerArray *, int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------------- ------ ------
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct         %    100
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct         %    100
  ---------------------------------------------------------------- ------ ------
```

데이터 레이아웃을 보면 구조체 안에 N개의 x 멤버가 연속적으로 나타나고, 바로 이어서 N개의 y 멤버가 연속적으로 나타난다. 따라서, 각 스레드에서 x멤버를 읽거나 저장할 때 bandwidth의 낭비가 없다. 따라서, global memory load/store 효율이 100%로 측정된다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [NVIDIA CUDA Documentation: Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)