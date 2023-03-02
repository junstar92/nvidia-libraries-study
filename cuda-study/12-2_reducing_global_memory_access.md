# Table of Contents

- [Table of Contents](#table-of-contents)
- [Reducing Global Memory Access](#reducing-global-memory-access)
- [Parallel Reduction with Shared Memory](#parallel-reduction-with-shared-memory)
- [Parallel Reduction with Unrolling](#parallel-reduction-with-unrolling)
- [Parallel Reduction with Dynamic Shared Memory](#parallel-reduction-with-dynamic-shared-memory)
- [Effiective Bandwidth](#effiective-bandwidth)
- [References](#references)

<br>

# Reducing Global Memory Access

Shared memory를 사용하는 이유 중 하나는 데이터를 on-chip에 캐싱하기 위해서이다. 이를 통해 커널 내에서 global memory access 횟수를 감소시킨다. [Avoiding Branch Divergence](/cuda-study/07_avoiding_branch_divergence.md)와 [Unrolling Loops](/cuda-study/08_unrolling_loops.md)에서 parallel reduction problem을 다루었는데, 이번 포스팅에서는 parallel reduction에 shared memory를 사용하는 커널을 구현하고 성능에 어떠한 영향을 미치는지 살펴본다.

> 해당 포스팅에서 사용되는 전체 코드는 [reduce_smem.cu](/code/cuda/reduce_integer/reduce_smem.cu)을 참조

# Parallel Reduction with Shared Memory

아래의 `reduceGmem` 커널을 baseline으로 살펴보자. 이 커널은 [Unrolling Loops](/cuda-study/08_unrolling_loops.md)에서 다루었던 커널이며 오직 global memory만을 사용하고 unrolling warp 기법이 적용되어 있다.

```c++
// unrolling warp + gmem
__global__
void reduceGmem(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;
    
    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

간단하게 해당 커널을 요약하자면, 4가지 부분으로 나누어서 설명할 수 있다. 먼저 연산이 처리되는 스레드 블록에 속하는 data chunk를 연산하기 위한 data offset은 다음과 같이 계산하며, 이는 global input에 상대적인 주소값을 가지고 있다.
```c++
int* in = g_in + blockDim.x * threadIdx.x;
```

그리고 32개의 요소가 남을 때까지 in-place reduction을 수행한다 (using global memory).
```c++
// in-place reduction and complete unroll
if (blockDim.x >= 1024 && tid < 512)
    in[tid] += in[tid + 512];
__syncthreads();

if (blockDim.x >= 512 && tid < 256)
    in[tid] += in[tid + 256];
__syncthreads();

if (blockDim.x >= 256 && tid < 128)
    in[tid] += in[tid + 128];
__syncthreads();

if (blockDim.x >= 128 && tid < 64)
    in[tid] += in[tid + 64];
__syncthreads();
```

그러고 난 뒤, 커널은 스레드 블록의 첫 번째 warp만을 사용하여 in-place reduction을 수행하는데, 여기에서는 `volatile` qualifier을 사용하여 unrolling warp 기법을 적용한다.
```c++
// unrolling warp
if (tid < 32) {
    volatile int* vmem = in;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
}
```

마지막으로, 스레드 블록에 할당된 input data chunk에 대한 total sum을 다시 global output memory에 저장한다.
```c++
if (tid == 0) g_out[blockIdx.x] = in[0];
```

> 포스팅에서 테스트하는 모든 커널들에 대한 입력 배열의 크기는 16M이며, 스레드 블록의 크기는 256 threads를 사용한다.

[reduce_smem.cu](/code/cuda/reduce_integer/reduce_smem.cu) 코드를 컴파일하고, `nsight system`으로 커널을 프로파일링해보면 아래와 같은 결과를 얻을 수 있다.
```
$ nvcc -O3 -o reduce_smem reduce_smem.cu
$ nsys profile --stats=true ./reduce_smem
...
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)      GridXYZ         BlockXYZ                          Name                      
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ---------------  --------------  -----------------------------------------------
     25.8          248,705          1  248,705.0  248,705.0   248,705   248,705          0.0  65536    1    1   256    1    1  reduceGmem(int *, int *, unsigned int)
```

약 0.248ms의 속도로 측정이 되며, 이 측정값을 baseline으로 설정하고 이후에 구현하는 커널들의 성능과 비교한다.

이제 shared memory를 사용하여 in-place reduction을 수행하는 `reduceSmem` 커널을 구현해보자. 기본적인 구현은 `reduceGmem`과 거의 동일하다. 다만 global memory input을 그대로 사용하는 것이 아닌 shared memory array `smem`을 사용한다. `smem`은 각 스레드 블록의 크기와 동일한 크기로 정적으로 선언된다 (`DIM = 256`).
```c++
__shared__ int smem[DIM];
```

커널 내에서 각 스레드 블록은 `smem`을 global input data chunk로 초기화를 먼저 해준다.
```c++
smem[tid] = in[tid];
__syncthreads();
```
그러고 난 뒤, in-place reduction을 `smem`을 사용하여 수행한다. `reduceSmem` 커널의 전체 구현은 다음과 같다.
```c++
// reduction using shared memory + unrolling warp
__global__
void reduceSmem(int* g_in, int* g_out, unsigned int const n)
{
    __shared__ int smem[DIM];
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x;

    // set to smem by each threads
    smem[tid] = in[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = smem[0];
}
```

위 커널 함수를 추가하고 프로파일링한 결과는 다음과 같다.
```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)      GridXYZ         BlockXYZ                          Name                      
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ---------------  --------------  -----------------------------------------------
     25.8          248,705          1  248,705.0  248,705.0   248,705   248,705          0.0  65536    1    1   256    1    1  reduceGmem(int *, int *, unsigned int)         
     20.8          201,153          1  201,153.0  201,153.0   201,153   201,153          0.0  65536    1    1   256    1    1  reduceSmem(int *, int *, unsigned int)
```
Shared memory를 사용하지 않은 `reduceGmem` 커널의 수행 시간은 0.248ms으로 측정되지만, shared memory를 사용하게 되면 0.201ms로 더 빠르게 측정된다.

Shared memory를 사용하는 이유 중 하나가 global memory access를 줄이는 것이라고 했으니, 실제로 줄어들었는지 `nsight compute`를 통해 측정해보자.
```
$ ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./reduce_smem
...
reduceGmem(int *, int *, unsigned int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------- --------------- -----------------
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                      sector         6,553,600
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                      sector         3,211,264
  ---------------------------------------------------------- --------------- -----------------
reduceSmem(int *, int *, unsigned int), Context 1, Stream 7
  Section: Command line profiler metrics
  ---------------------------------------------------------- --------------- -----------------
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                      sector         2,097,152
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                      sector            65,536
  ---------------------------------------------------------- --------------- -----------------
```
측정 결과, 상당한 양의 global memory access가 감소한 것을 볼 수 있다.

# Parallel Reduction with Unrolling

[Unrolling Loops](/cuda-study/08_unrolling_loops.md)에서 다루었던 unrolling loop 기법을 적용해보자. 커널에서의 unrolling loop는 하나의 스레드 블록이 여러 개의 데이터 블록을 처리하도록 하는 기법을 의미한다. 여기서는 하나의 스레드 블록의 4개의 데이터 블록을 처리하도록 하여, 커널의 성능을 향상시킨다.

Unrolling을 적용하여 기대할 수 있는 이점은 아래와 같이 요약할 수 있다.
- 스레드에 더 많은 parallel I/O를 노출시켜 global memory throughput을 증가시킬 수 있음
- Global memory store transaction이 1/4로 감소
- 전체적인 커널 성능 향상

먼저 global memory만을 사용하는 unroll 커널 `reduceGmemUnroll` 구현은 다음과 같다.
```c++
// unrolling 4 thread blocks + unrolling warp + gmem
__global__
void reduceGmemUnroll(int* g_in, int* g_out, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int* in = g_in + blockDim.x * blockIdx.x * 4;
    
    // unrolling 4 data blocks
    if (idx + blockDim.x * 3 < n) {
        int sum = 0;
        sum += g_in[idx + blockDim.x];
        sum += g_in[idx + blockDim.x * 2];
        sum += g_in[idx + blockDim.x * 3];
        g_in[idx] += sum;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512)
        in[tid] += in[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        in[tid] += in[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        in[tid] += in[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        in[tid] += in[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = in;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = in[0];
}
```

그리고 shared memory를 사용하면서 unrolling 기법을 적용한 `reduceSmemUnroll` 커널 구현은 다음과 같다.
```c++
// shared memory + unrolling 4 thread blocks + unrolling warp
__global__
void reduceSmemUnroll(int* g_in, int* g_out, unsigned int const n)
{
    __shared__ int smem[DIM];
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    
    // unrolling 4 blocks
    int tmp_sum = 0;
    if (idx + blockDim.x * 3 <= n) {
        tmp_sum += g_in[idx];
        tmp_sum += g_in[idx + blockDim.x];
        tmp_sum += g_in[idx + blockDim.x * 2];
        tmp_sum += g_in[idx + blockDim.x * 3];
    }
    smem[tid] = tmp_sum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = smem[0];
}
```

> Unrolling 기법에 대한 자세한 내용은 [Unrolling Loops](/cuda-study/08_unrolling_loops.md)를 참조

당연하지만, 각 스레드가 4개의 데이터 요소를 처리하기 때문에 커널을 실행할 때 그리드의 크기를 1/4로 감소시켜주어야 한다.
```c++
reduceSmemUnroll<<<grid.x / 4, block>>>(d_in, d_out, num_elements);
```

이들 커널에 대한 프로파일링 결과는 다음과 같다.
```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)      GridXYZ         BlockXYZ                          Name                      
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ---------------  --------------  -----------------------------------------------
     25.8          248,705          1  248,705.0  248,705.0   248,705   248,705          0.0  65536    1    1   256    1    1  reduceGmem(int *, int *, unsigned int)         
     20.8          201,153          1  201,153.0  201,153.0   201,153   201,153          0.0  65536    1    1   256    1    1  reduceSmem(int *, int *, unsigned int)     
     12.5          120,673          1  120,673.0  120,673.0   120,673   120,673          0.0  16384    1    1   256    1    1  reduceGmemUnroll(int *, int *, unsigned int)   
     10.1           97,888          1   97,888.0   97,888.0    97,888    97,888          0.0  16384    1    1   256    1    1  reduceSmemUnroll(int *, int *, unsigned int)
```

단순히 shared memory만을 사용했을 때는 0.201ms가 걸렸지만, unrolling 기법을 적용하면 0.097ms로 약 2배 빠르게 측정된다.

> 위 결과에서 주목할만한 것은 `reduceGmemUnroll`의 성능이다. Shared memory를 전혀 사용하지 않았지만, unrolling 기법을 적용한 것만으로 shared memory를 사용한 `reduceSmem`보다 약 1.6배 빠른 속도를 보여준다.

`reduceSmem`과 `reduceSmemUnroll`의 global memory transaction의 수를 측정해보면 아래와 같이 측정된다.
```
  reduceSmem(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    -------------------------------------------------------- --------------- -----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                    sector         2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                    sector            65,536
    -------------------------------------------------------- --------------- -----------------

  reduceSmemUnroll(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    -------------------------------------------------------- --------------- -----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                    sector         2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                    sector            16,384
    -------------------------------------------------------- --------------- -----------------
```

위 결과를 통해 global memory store transaction의 수가 1/4로 감소했다는 것을 볼 수 있다.

이번에는 global memory throughput을 측정해보자. `nsight compute`를 통해 아래의 커맨드로 global memory throughput을 측정할 수 있다.
```
$ ncu --metrics=l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second ./reduce_smem
...
  reduceGmem(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- --------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second         755.63
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second         370.26
    ---------------------------------------------------------------------- --------------- --------------

  reduceSmem(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- --------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second         304.02
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second           9.50
    ---------------------------------------------------------------------- --------------- --------------

  reduceGmemUnroll(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- --------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second         989.55
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second         351.55
    ---------------------------------------------------------------------- --------------- --------------

  reduceSmemUnroll(int *, int *, unsigned int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- --------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second         699.52
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second           5.46
    ---------------------------------------------------------------------- --------------- --------------
```

`reduceSmem` 커널과 `reduceSmemUnroll` 커널을 비교헀을 때, global memory load throughput은 약 2.3배 상승(304.02->699.52 GB/s)했고, global memory store throughput은 약 1.7배 하락(9.50->5.46 GB/s)했다는 것을 알 수 있다. Load throughput이 증가한 이유는 동시에 수행되는 load request 수가 더 많기 때문이고, store throughput이 감소한 이유는 store request의 절대적인 수치가 감소했기 때문이다.

# Parallel Reduction with Dynamic Shared Memory

Shared memory를 사용하는 parallel reduction 커널을 구현할 때, shared memory를 동적으로 할당하도록 구현할 수 있다. 구현은 `reduceSmem`이나 `reduceSmemUnroll`과 같으며, 단순히 아래와 같이 shared memory를 선언해주기만 하면 된다.
```c++
extern __shared__ int smem[];
```

그리고, 커널을 실행할 때, execution configuration에 동적으로 할당할 shared memory의 크기를 지정해주면 된다.
```c++
reduceSmemUnrollDyn<<<grid.x / 4, block, DIM * sizeof(int)>>>(d_in, d_out, num_elements);
```

커널(`reduceSmemDyn`, `reduceSmemUnrollDyn`) 코드 구현은 [reduce_smem.cu](/code/cuda/reduce_integer/reduce_smem.cu)에서 확인할 수 있다.

> [Data Layout of Shared Memory](/cuda-study/12-1_data_layout_of_shared_memory.md)에서도 언급했지만, 동적으로 shared memory를 할당할 때와 정적으로 할당할 때의 성능 차이는 확인이 되지 않는다. 측정 시, 동적으로 할당할 때가 더 빠르게 측정될 때도 있다.

# Effiective Bandwidth

Reduction 커널은 memory bandwidth에 의해 제한되므로, reduction 커널을 평가하는 적절한 performance metric으로 `effective bandwidth`를 사용할 수 있다. Effective bandwidth는 커널이 완전히 수행되는 시간 동안 처리된 I/O의 크기로 계산할 수 있다. 식을 표현하면 다음과 같다.

$$ \text{effective bandwidth = (bytes read + bytes written)} \div (\text{time(sec) } \times 10^9) \text{ } GB/s $$

아래 표는 위에서 구현한 커널들의 대한 성능을 요약한 결과이다.

|Kernels|Elapsed Time (ms)|Read Data Elements|Write Data Elements|Total Bytes|Bandwidth (GB/s)|
|:--|--|--|--|--|--|
|`reduceGmem`|0.248|16777216|65536|67371008|271.66|
|`reduceSmem`|0.201|16777216|65536|67371008|335.18|
|`reduceGmemUnroll`|0.120|16777216|16384|67174400|559.79|
|`reduceSmemUnroll`|0.097|16777216|16384|67174400|692.52|

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher