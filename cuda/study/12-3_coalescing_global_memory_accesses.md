# Table of Contents

- [Table of Contents](#table-of-contents)
- [Coalescing Global Memory Access](#coalescing-global-memory-access)
- [Baseline Transpose Kernel](#baseline-transpose-kernel)
- [Matrix Transpose with Shared Memory](#matrix-transpose-with-shared-memory)
- [Matrix Transpose with Padded Shared Memory](#matrix-transpose-with-padded-shared-memory)
- [Matrix Transpose with Unrolling](#matrix-transpose-with-unrolling)
- [References](#references)

# Coalescing Global Memory Access

Shared memory를 사용하면 non-coalesced global memory access를 피할 수 있다. Matrix transpose가 이에 해당한다. 일반적으로 read operation은 자연스럽게 coalesced access로 처리할 수 있지만, write operation은 strided access로 처리된다. [Matrix Transpose Problem](/cuda/study/11-1_matrix_transpose_problem.md)에서 strided access는 최악의 메모리 액세스 패턴이라는 것을 살펴봤다. Shared memory를 사용하면 transpose operation은 shared memory를 처리하고, global memory에 write하는 작업을 coalesced access로 처리할 수 있다.

# Baseline Transpose Kernel

> 전체 구현 코드는 [transpose_smem.cu](/cuda/code/matrix_transpose/transpose_smem.cu)를 참조

먼저 [Matrix Transpose Problem](/cuda/study/11-1_matrix_transpose_problem.md)에서 naive하게 구현한 행렬 전치 커널로 시작한다.

```c++
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

위 커널 구현에서 `ix`는 이 커널의 2D thread configuration의 innermost 차원에 속하므로, global memory read operation은 warp 내에서 coalesced access로 처리된다. 반면 global memory write operation에서는 인접한 스레드들끼리 strided acecss로 처리된다는 것을 알 수 있다. `transposeNaiveRow` 커널은 이번 포스팅에서 살펴볼 커널들의 성능 중 lower bound이다.

아래의 `copyRow` 커널은 write operation을 coalesced access로 처리되도록 구현한 것이다.
```c++
__global__
void copyRow(float* out, float const* in, int const nx, int const ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}
```

이 커널은 read/write operation이 모두 coalesced access로 처리되고, 우리가 구현할 행렬 전치 커널과 동일한 크기의 I/O를 수행하므로 위 커널의 성능이 대략적인 upper bound가 된다.

여기서 구현한 모든 커널들에 대한 입력 행렬의 크기는 `4096x4096`이며, 스레드 블록의 차원은 `32x16`이다 ([transpose_smem.cu](/cuda/code/matrix_transpose/transpose_smem.cu) 참조).

두 커널 함수에 대해 실행 결과는 다음과 같다.
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 4096 x 4096
CopyGmemRow            elapsed time: 0.199680 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 672.164124 GB/s
NaiveGmemRow           elapsed time: 0.616448 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 217.727585 GB/s
```

당연히 naive한 구현 커널인 `transposeNaiveRow`가 `copyRow`보다 훨씬 느리다. 이는 `transposeNaiveRow`에서 write operation은 4096 stride 크기의 strided access로 처리되기 때문이다. 결과적으로 한 warp 내에서 store memory operation은 32번의 memory transaction(32-bytes)으로 처리된다. 이는 `nsight compute`로 global memory load/store transaction 수를 측정해보면 알 수 있다.
```
$ sudo ncu --metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./transpose_smem
...
  copyRow(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ----------------------------------------------------------------- --------------- ----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             sector        2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                             sector        2,097,152
    ----------------------------------------------------------------- --------------- ----------------

  transposeNaiveRow(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ----------------------------------------------------------------- --------------- ----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             sector        2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                             sector       16,777,216
    ----------------------------------------------------------------- --------------- ----------------
...
```

`copyRow` 커널의 경우, 한 warp 내에서 처리하는 요소가 `4 bytes x 32 = 128 bytes`이므로 4개의 32-bytes memory transaction으로 처리된다. 따라서, 커널에서 발생되는 모든 global memory load transaction은 `4 x 128 x 256 x 16 = 2097152`로 계산된다.

반면 `transposeNaiveRow`에서 store operation은 strided access로 처리되므로, 각 스레드에서 처리되는 데이터 요소들은 각각의 memory transaction으로 처리된다. 따라서, 한 warp 내에서 32개의 데이터를 처리하므로, 32번의 memory transaction이 발생하게 된다. 따라서, 커널 내에서 발생되는 모든 global memory store transaction의 수는 `32 x 128 x 256 x 16 = 16777216`으로 계산된다.

[transpose_smem.cu](/cuda/code/matrix_transpose/transpose_smem.cu)에서는 read operation은 strided access로 처리하고 write operation을 coalesced access로 처리하는 `transposeNaiveCol` 커널도 구현했고, 하나의 스레드 블록이 4개의 데이터 블록을 처리하도록 unrolling 기법을 적용한 `transposeUnroll4Row`, `transposeUnroll4Col` 커널도 구현되어 있다. 커널들의 성능은 아래와 같이 측정된다.
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 4096 x 4096
CopyGmemRow            elapsed time: 0.199680 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 672.164124 GB/s
CopyGmemCol            elapsed time: 0.644096 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 208.381546 GB/s
NaiveGmemRow           elapsed time: 0.616448 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 217.727585 GB/s
NaiveGmemCol           elapsed time: 0.338624 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 396.362122 GB/s
NaiveGmemUnroll4Row    elapsed time: 0.619616 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 216.614380 GB/s
NaiveGmemUnroll4Col    elapsed time: 0.301760 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 444.783051 GB/s
```

Unrolling 기법을 적용하면 조금 더 빨라지지만, 여전히 strided access로 인해 필요한 memory transaction의 수는 동일하게 측정되는 것을 `nsight compute`를 통해 알 수 있다.
```
  transposeUnroll4Row(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ----------------------------------------------------------------- --------------- ----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             sector        2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                             sector       16,777,216
    ----------------------------------------------------------------- --------------- ----------------

  transposeUnroll4Col(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ----------------------------------------------------------------- --------------- ----------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             sector       16,777,215
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                             sector        2,097,152
    ----------------------------------------------------------------- --------------- ----------------
```

# Matrix Transpose with Shared Memory

2D shared memory를 사용해 원본 행렬로부터 데이터를 캐싱하면, strided global memory access를 해결할 수 있다.

기본 구현 방법은 다음과 같다. 먼저, 원본 행렬로부터 한 블록에 해당하는 데이터(global memory)를 row-order로 읽어서 shared memory에 row-order로 저장한다. 그리고, shared memory로부터 column-order로 데이터를 읽고, 그 데이터를 transposed matrix에 row-order로 저장한다. 이 구현 방법을 그림으로 표현하면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJxo5B%2Fbtr1HhzQkR3%2FMK1D6QidrdXsHWunI0L9TK%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

위와 같이 구현하게 되면 사실 shared memory에서 bank conflict는 발생한다. 하지만 non-coalesced global memory access보다 성능은 더 좋다.

위 구현대로 커널을 구현한 코드는 아래와 같다.
```c++
__global__
void transposeSmem(float* out, float const* in, int const nx, int const ny)
{
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = blockDim.x * threadIdx.y + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    if (ix < nx && iy < ny) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}
```

위 코드 구현은 아래의 단계로 수행된다.
1. 스레드 블록 내 각 warp는 global memory에 저장된 원본 행렬의 한 블록으로부터 하나의 row를 coalesced access로 read한다.
2. 스레드 블록 내 각 warp는 1에 읽은 한 row의 데이터를 row-major order로 shared memory에 저장한다. Shared memory write에서는 bank conflict가 발생하지 않는다.
3. 스레드 블록 내의 모든 read/write operation을 동기화한다.
4. 스레드 블록 내 각 warp는 2D shared memory array로부터 하나의 column을 읽는다. Column-major order read 이므로 요청된 메모리들이 동일한 bank에 있으므로 bank conflict가 발생한다.
5. Shared memory로 부터 읽은 데이터를 global memory에 저장된 transposed matrix에 coalesced access로 write한다.

Global memory와 shared memory로부터 올바른 데이터를 전달하기 위해서, 각 스레드는 몇 가지 인덱스들을 계산해야 한다. 주어진 스레드에서 먼저 원본 행렬 기준 데이터 좌표를 아래와 같이 계산한다. `ti`는 global memory original matrix 인덱스이다. `ix`가 스레드 블록의 innermost 차원이므로, 각 warp는 `ti`를 통해 coalesced read를 수행한다.
```c++
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
ti = iy * nx + ix;
```

결과 행렬인 transpose matrix에서의 좌표도 유사하게 계산하는데, 방법이 조금 다르다. 먼저, transpose matrix에서 블록의 오프셋을 계산할 때 주의해야 하는데, `blockDim`과 `blockIdx`의 내부 차원이 서로 스왑된다. Transpose matrix에서는 `x` 차원 값이 열 좌표를 계산할 때 사용되고, `y` 차원 값이 행 좌표를 계산할 때 사용된다.

또한, `icol`과 `irow`라는 새로운 변수를 사용하여 transpose matrix의 블록 내 좌표로 사용한다.
```c++
// thread index in transposed block
unsigned int bidx, irow, icol;
bidx = blockDim.x * threadIdx.y + threadIdx.x;
irow = bidx / blockDim.y;
icol = bidx % blockDim.y;
```

이렇게 계산된 `icol`과 `irow`를 가지고 transposed matrix 내의 global index를 아래와 같이 계산한다.
```c++
// coordinate in transposed matrix
ix = blockDim.y * blockIdx.y + icol;
iy = blockDim.x * blockIdx.x + irow;

// linear global memory index for transposed matrix
to = iy * ny + ix;
```

아래 그림은 위의 좌표 계산을 시각화한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FM7eLi%2Fbtr1KQO0xly%2FwB3NxSVCDT2ygp8ygaeRq1%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

계산된 오프셋을 사용하여 각 warp 내 스레드들은 global memory로부터 연속된 데이터를 읽어서, 2D shared memory `tile`의 행에 저장한다 (coalesced global read access, bank conflict-free). 
```c++
tile[threadIdx.y][threadIdx.x] = in[ti];
```

그리고, 각 warp 내 스레드들은 shared memory `tile`로부터 열 데이터를 읽어서 global memory에 이 데이터를 연속적인 위치에 write한다 (bank conflict, coalesced global write access).
```c++
out[to] = tile[icol][irow];
```

실행 결과, 성능은 다음과 같이 측정된다.
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 4096 x 4096
CopyGmemRow            elapsed time: 0.199680 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 672.164124 GB/s
CopyGmemCol            elapsed time: 0.644096 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 208.381546 GB/s
NaiveGmemRow           elapsed time: 0.616448 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 217.727585 GB/s
NaiveGmemCol           elapsed time: 0.338624 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 396.362122 GB/s
NaiveGmemUnroll4Row    elapsed time: 0.619616 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 216.614380 GB/s
NaiveGmemUnroll4Col    elapsed time: 0.301760 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 444.783051 GB/s
transposeSmem          elapsed time: 0.263744 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 508.893951 GB/s
```

Unrolling이 적용된 global memory 커널보다 더 빠른 속도를 보여준다.

`nsight compute`로 측정한 global memory load/store transaction의 수는 다음과 같다.
```
  transposeSmem(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ----------------------------------------------------------------- --------------- ---------------
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                             sector       2,097,152
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                             sector       2,097,152
    ----------------------------------------------------------------- --------------- ---------------
```

Global memory만을 사용했을 때보다 global store transaction의 수가 1/8로 감소한 것을 확인할 수 있다.

<br>

# Matrix Transpose with Padded Shared Memory

방금 구현한 `transposeSmem` 커널에서는 shared memory로부터 값을 load할 때 bank conflict가 발생한다. 그 결과 필요한 것보다 더 많은 shared memory transaction이 발생하게 된다. 이 또한 `nsight compute`를 통해 측정할 수 있는데, 아래의 커맨드를 통해 측정한 `transposeSmem`의 shared memory load/store transaction의 수는 다음과 같다.
```
$ sudo ncu --metrics=l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./transpose_smem

  transposeSmem(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    --------------------------------------------------------------- ---------- ----------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                              8,484,348
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                551,904
    --------------------------------------------------------------- ---------- ----------------
```

> 매번 측정할 때마다 약간씩 다르다. 단, shared memory store transaction보다 load transaction이 월등히 많다는 것을 볼 수 있다.

[Shared Memory: Memory Padding](/cuda/study/12_shared_memory.md#memory-padding)에서 살펴봤듯이, shared memory에 padding을 추가하면 bank conflict를 해결할 수 있다. Static shared memory에서는 다른 코드 변경없이 shared memory array를 선언할 때 innermost 차원에 padding만 추가해주면 된다.
```c++
__shared__ float tile[BDIMY][BDIMX + PADDING];
```

`32x16` 스레드 블록으로 실행되므로, padding의 크기를 2로 지정해야 bank conflict를 완전히 제거할 수 있다.

전체 커널 구현은 다음과 같다.
```c++
__global__
void transposeSmemPad(float* out, float const* in, int const nx, int const ny)
{
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX + PADDING];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = blockDim.x * threadIdx.y + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    if (ix < nx && iy < ny) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}
```

실행 및 성능 결과는 다음과 같다.
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 4096 x 4096
CopyGmemRow            elapsed time: 0.199680 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 672.164124 GB/s
CopyGmemCol            elapsed time: 0.644096 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 208.381546 GB/s
NaiveGmemRow           elapsed time: 0.616448 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 217.727585 GB/s
NaiveGmemCol           elapsed time: 0.338624 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 396.362122 GB/s
NaiveGmemUnroll4Row    elapsed time: 0.619616 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 216.614380 GB/s
NaiveGmemUnroll4Col    elapsed time: 0.301760 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 444.783051 GB/s
transposeSmem          elapsed time: 0.263744 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 508.893951 GB/s
transposeSmemPad       elapsed time: 0.241600 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 555.536926 GB/s
```

Padding 없이 shared memory만을 사용헀을 때는 약 0.264ms가 걸렸지만, padding을 추가하면 약 0.242ms가 걸리게 된다.

Shared memory load/store transaction 측정 결과는 다음과 같다.
```
  transposeSmem(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------- ---------- ------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                 8,482,701
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   552,051
    ---------------------------------------------------------------- ---------- ------------------

  transposeSmemPad(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------- ---------- ------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                   551,911
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   547,747
    ---------------------------------------------------------------- ---------- ------------------
```

Bank conflict가 제거되어 `transposeSmem`보다 `transposeSmemPad` 커널의 shared memory load transaction의 수가 상당히 감소한 것을 볼 수 있다. 이론상 load 및 store가 1:1 비율이어야 하지만, 정확히 1:1로는 측정되지 않았다.

<br>

# Matrix Transpose with Unrolling

이번에는 shared memory에 unrolling 기법을 적용한다. 구현한 커널은 하나의 스레드 블록이 두 개의 데이터 블록을 처리하며, 이를 통해 더 많은 동시 in-flight load/store를 통해 device memory bandwidth 활용률을 향상시킬 수 있다.

커널의 구현은 다음과 같으며, memory padding이 없는 버전과 있는 버전 각각 따로 구현하였다.
```c++
// 8 transpose kernel: read by rows, write by columns + using shared memory + unrolling 2 thead blocks
__global__
void transposeSmemUnroll2(float* out, float const* in, int const nx, int const ny)
{
    // static 1D shared memory
    __shared__ float tile[BDIMY * BDIMX * 2];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to, ix2, iy2;
    ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = blockDim.x * threadIdx.y + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix2 = blockDim.y * blockIdx.y + icol;
    iy2 = 2 * blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny) {
        // load two rows from global memory to shared memory
        unsigned int row_idx = 2 * blockDim.x * threadIdx.y + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = 2 * blockDim.x * icol + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

// 9 transpose kernel: read by rows, write by columns + using shared memory + unrolling 2 thead blocks + memory padding
__global__
void transposeSmemUnroll2Pad(float* out, float const* in, int const nx, int const ny)
{
    // static 1D shared memory
    __shared__ float tile[BDIMY * (BDIMX * 2 + PADDING)];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to, ix2, iy2;
    ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = blockDim.x * threadIdx.y + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix2 = blockDim.y * blockIdx.y + icol;
    iy2 = 2 * blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny) {
        // load two rows from global memory to shared memory
        unsigned int row_idx = (2 * blockDim.x + PADDING) * threadIdx.y + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = (2 * blockDim.x + PADDING) * icol + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}
```

두 개의 데이터 블록을 처리하기 때문에, 인덱스 계산이 조금 달라진다. 먼저, global memory array index와 original matrix 좌표 값은 아래와 같이 계산한다.
```c++
ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
ti = iy * nx + ix;
```

그리고 설정한 스레드 블록의 크기는 32x16이므로, 한 스레드 블록에서 처리되는 데이터 크기는 (32+32)x16이다. 이를 그림으로 나타내면 아래와 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fq6D6y%2Fbtr1EfaZPyB%2FP6JS7TnIph01MpDoNBEb5k%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

Shared memory에서 transposed block에서의 스레드 인덱스 값은 아래와 같이 계산할 수 있고,
```c++
bidx = blockDim.x * threadIdx.y + threadIdx.x;
irow = bidx / blockDim.y;
icol = bidx % blockDim.y;
```

shared memory array `tile`이 1D이므로, 아래와 같이 shared memory의 `row_idx`, `col_idx`를 각각 구할 수 있다.
```c++
unsigned int row_idx = (2 * blockDim.x + PADDING) * threadIdx.y + threadIdx.x;
unsigned int col_idx = (2 * blockDim.x + PADDING) * icol + irow;
```

두 커널을 포함하여 실행한 결과는 다음과 같다.
```
> Matrix transpose at device 0: NVIDIA GeForce RTX 3080
> with matrix 4096 x 4096
CopyGmemRow            elapsed time: 0.199680 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 672.164124 GB/s
CopyGmemCol            elapsed time: 0.644096 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 208.381546 GB/s
NaiveGmemRow           elapsed time: 0.616448 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 217.727585 GB/s
NaiveGmemCol           elapsed time: 0.338624 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 396.362122 GB/s
NaiveGmemUnroll4Row    elapsed time: 0.619616 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 216.614380 GB/s
NaiveGmemUnroll4Col    elapsed time: 0.301760 ms <<< grid(32,256) block(32,16)>>> effective bandwidth: 444.783051 GB/s
transposeSmem          elapsed time: 0.263744 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 508.893951 GB/s
transposeSmemPad       elapsed time: 0.241600 ms <<< grid(128,256) block(32,16)>>> effective bandwidth: 555.536926 GB/s
transposeSmemUnroll    elapsed time: 0.215744 ms <<< grid(64,256) block(32,16)>>> effective bandwidth: 622.115662 GB/s
transposeSmemUnrollPad elapsed time: 0.212896 ms <<< grid(64,256) block(32,16)>>> effective bandwidth: 630.437988 GB/s
```

Unrolling 기법이 적용되지 않은 커널들보다 더 좋은 성능을 보여주고 있다.

커널들의 성능 차이는 `nsight compute`를 통해 device memory read/write throughput을 측정하여 비교해볼 수도 있다. 실행 커맨드와 결과는 다음과 같다.
```
$ sudo ncu --metrics=dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second ./transpose_smem
...
  transposeNaiveRow(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second           92.30
    dram__bytes_write.sum.per_second                                          Gbyte/second           89.35
    ---------------------------------------------------------------------- --------------- ---------------

  transposeNaiveCol(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          210.78
    dram__bytes_write.sum.per_second                                          Gbyte/second          202.95
    ---------------------------------------------------------------------- --------------- ---------------

  transposeUnroll4Row(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second           93.27
    dram__bytes_write.sum.per_second                                          Gbyte/second           93.68
    ---------------------------------------------------------------------- --------------- ---------------

  transposeUnroll4Col(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          229.11
    dram__bytes_write.sum.per_second                                          Gbyte/second          220.75
    ---------------------------------------------------------------------- --------------- ---------------

  transposeSmem(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          251.47
    dram__bytes_write.sum.per_second                                          Gbyte/second          243.01
    ---------------------------------------------------------------------- --------------- ---------------

  transposeSmemPad(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          284.76
    dram__bytes_write.sum.per_second                                          Gbyte/second          275.18
    ---------------------------------------------------------------------- --------------- ---------------

  transposeSmemUnroll2(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          320.09
    dram__bytes_write.sum.per_second                                          Gbyte/second          309.26
    ---------------------------------------------------------------------- --------------- ---------------

  transposeSmemUnroll2Pad(float *, const float *, int, int), Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ---------------
    dram__bytes_read.sum.per_second                                           Gbyte/second          332.32
    dram__bytes_write.sum.per_second                                          Gbyte/second          321.21
    ---------------------------------------------------------------------- --------------- ---------------
```

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher