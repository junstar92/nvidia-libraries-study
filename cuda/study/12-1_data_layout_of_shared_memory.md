# Table of Contents

- [Table of Contents](#table-of-contents)
- [Checking the Data Layout of Shared Memory](#checking-the-data-layout-of-shared-memory)
- [Square Shared Memory](#square-shared-memory)
  - [Accessing Row-Major vs. Column-Major](#accessing-row-major-vs-column-major)
  - [Writing Row-Major and Reading Column-Major](#writing-row-major-and-reading-column-major)
  - [Dynamic Shared Memory](#dynamic-shared-memory)
  - [Padding Statically Declared Shared Memory](#padding-statically-declared-shared-memory)
  - [Padding Dynamically Declared Shared Memory](#padding-dynamically-declared-shared-memory)
  - [Comparing the Performance of the Square Shared Memory Kernels](#comparing-the-performance-of-the-square-shared-memory-kernels)
- [Rectangular Shared Memory](#rectangular-shared-memory)
  - [Accessing Row-Major vs. Accessing Column-Major](#accessing-row-major-vs-accessing-column-major)
  - [Writing Row-Major and Reading Column-Major](#writing-row-major-and-reading-column-major-1)
  - [Dynamically Declared Shared Memory](#dynamically-declared-shared-memory)
  - [Padding Statically Declared Shared Memory](#padding-statically-declared-shared-memory-1)
  - [Padding Dynamically Declared Shared Memory](#padding-dynamically-declared-shared-memory-1)
  - [Comparing the Performance of the Rectangular Shared Memory Kernels](#comparing-the-performance-of-the-rectangular-shared-memory-kernels)
- [References](#references)

<br>

# Checking the Data Layout of Shared Memory

Shared memory를 효율적으로 사용하는 방법을 이해하기 위해 이번 포스팅에서는 shared memory를 사용하는 몇 가지 간단한 예제를 통해 다음의 주제들에 대해 살펴본다.

- Square Arrays vs. Rectangular Arrays
- Row-major accesses vs. Column-major accesses
- Static vs. Dynamic shared memory declarations
- File-scope vs. Kernel-scope shared memory
- Memory padding vs. No memory padding

Shared memory를 사용하여 커널의 성능을 최적화하려면 [Introducing CUDA Shared Memory](12_shared_memory.md)에서 살펴본 아래의 두 가지 개념에 주목해야 한다.

- Mapping data elements across memory banks
- Mapping from thread index to shared memory offset

위의 두 가지 개념만 명확하다면 bank conflict를 피하고 shared memory이 이점을 최대한 활용할 수 있는 커널을 설계할 수 있다.

<br>

# Square Shared Memory

간단한 방식으로 정사각 차원의 global data를 shared memory를 사용하여 캐싱할 수 있다. 정사각 배열의 차원은 단순하기 때문에 2D thread 인덱스로부터 1D memory offset을 쉽게 계산할 수 있다. 아래 그림은 row-major order로 저장된 각 차원의 크기가 32개의 요소인 shared memory tile을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FxFCqD%2Fbtr1beRrrmj%2F3VywQF7wz2zJ1kYIKkbe7k%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

2D shared memory 변수는 다음과 같이 정적으로 선언할 수 있다.
```c++
__shared__ int tile[N][N];
```

이 shared memory tile은 정사각형이므로, 2D 스레드 블록으로부터 `x` 또는 `y` 차원으로 이웃하는 스레드가 이웃하는 요소에 액세스하도록 할 수 있다.
```c++
tile[threadIdx.y][threadIdx.x]
tile[threadIdx.x][threadIdx.y]
```

위의 두 가지 방법 중 어떤 것이 더 성능이 좋을까?

이에 대한 답을 얻으려면 스레드들이 어떻게 shared memory bank에 매핑되는지를 살펴봐야 한다. [Bank Conflicts](12_shared_memory.md#bank-conflict)로부터 같은 warp 내의 스레드들이 서로 다른 bank에 액세스할 때 가장 최적이라고 했다. 동일한 warp 내 스레드들은 `threadIdx.x`의 연속적인 값으로부터 식별될 수 있다. 서로 다른 bank에 속한 shared memory 내의 요소들도 연속적으로 저장되어 있다. 따라서, 스레드들이 연속된 `threadIdx.x` 값을 따라 shared memory에 연속적인 위치에 액세스하는 것이 가장 최선이다. 결과적으로 첫 번째 액세스 패턴(`tile[threadIdx.y][threadIdx.x]`)가 더 적은 bank conflict가 발생되고 더 좋은 성능을 보여줄 것이다.

## Accessing Row-Major vs. Column-Major

> 전체 코드는 [smem_square.cu](/cuda/code/shared_memory_access/smem_square.cu)를 참조

각 차원에 32개의 스레드가 있는 하나의 그리드로 실행되는 예제 커널들을 살펴보자. [smem_square.cu](/cuda/code/shared_memory_access/smem_square.cu) 코드에서는 블록의 차원을 아래의 매크로를 통해 정의한다.
```c++
#define BDIMX 32
#define BDIMY 32
```

그리고 커널 실행을 위한 execution configuration은 다음과 같이 정의할 수 있다. 여기서는 커널을 하나의 스레드 블록으로만 실행한다.
```c++
dim3 block(BDIMX, BDIMY);
dim3 grid(1, 1);
```

살펴볼 예제 커널들은 다음의 두 가지 간단한 연산을 수행한다.

- Row-major order로 global thread index 값을 2D shared memory array에 쓴다
- Shared memory로부터 row-major order로 저장된 값을 읽고, 그 값을 global memory에 저장한다

먼저, 커널 내에서 2D shared memory를 다음과 같이 정적으로 선언할 수 있다.
```c++
__shared__ int tile[BDIMY][BDIMX];
```

그리고, 커널 내에서 각 스레드의 global thread index는 아래와 같이 계산할 수 있다 (오직 하나의 스레드 블록만 실행되므로 아래와 같이 간단하다).
```c++
unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;
```

계산한 global thread index 값을 row-major order로 shared memory tile에 저장하는 것은 아래의 코드로 수행한다.
```c++
tile[threadIdx.y][threadIdx.x] = idx;
```

각 스레드에서 shared memory에 인덱스 값을 저장하고 난 뒤에는 `__synchthreads()`를 통해 동기화하는 것이 필요하다. 모든 스레드들이 이 동기화 지점에 도달하면, 아래 코드를 통해 shared memory에 저장된 값을 row-major order로 global memory에 할당한다.
```c++
out[idx] = tile[threadIdx.y][threadIdx.x];
```

Row-major order로 read/write하는 커널의 전체 구현은 아래와 같다.
```c++
__global__
void setRowReadRow(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}
```
위 커널에서는 3개의 memory operation이 수행된다.

- One store operation on shared memory
- One load operation on shared memory
- One store operation on global memory

동일한 warp 내 스레드들은 연속적인 `threadIdx.x` 값을 가지고 `threadIdx.x`를 사용하여 shared memory array인 `tile`의 innermost 차원을 인덱싱하므로 bank conflict가 발생하지 않는다. 즉, warp 내 스레드들은 전부 서로 다른 bank에 있는 메모리 주소를 요청한다.

하지만, 위 커널에서 `threadIdx.x`와 `threadIdx.y`를 서로 바꾸면, warp 내에서 메모리 액세스는 **column-major order**로 수행된다. 결과적으로 warp 내 모든 스레드들은 같은 bank에 있는 서로 다른 메모리 주소를 요청하게 되어 32-way bank conflict가 발생한다.

> 만약 shared memory bank size가 8-byte라면 16-way bank conflict가 발생한다.

Column-major order로 동작하는 커널 구현은 다음과 같다.
```c++
__global__
void setColReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```

이제 위의 두 커널 `setRowReadRow`와 `setColReadCol` 커널의 성능을 비교해보자. 성능 비교는 `nsight system`으로 수행했다.

```
$ nsys profile --stats=true ./smem_square
> At device 0: NVIDIA GeForce RTX 3080 with Bank Size: 4-Byte
<<< grid (1,1) block (32,32)>>>
...
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)     GridXYZ         BlockXYZ                Name           
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------  --------------  --------------------------
     23.5            3,648          1   3,648.0   3,648.0     3,648     3,648          0.0     1    1    1    32   32    1  setColReadCol(int *)      
     15.6            2,432          1   2,432.0   2,432.0     2,432     2,432          0.0     1    1    1    32   32    1  setRowReadRow(int *)      
```

위 결과는 RTX3080에서 4-byte shared memory access mode일 때의 결과이다. 예상할 수 있듯이 row-major order로 메모리를 액세스하는 커널 `setRowReadRow`의 성능이 더 좋다.

성능이 더 좋은 이유는 `nsight compute`를 통해 두 커널의 shared memory transaction을 프로파일링해보면 쉽게 알 수 있다. Shared memory load/store transaction의 수는 `nsight compute`의 아래 두 메트릭을 통해 측정할 수 있다.

- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`

측정 결과는 다음과 같다.
```
$ ncu --metrics=l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./smem_square
setColReadCol(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  -------------------------------------------------------- --------------- -------------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                             1,024
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                             1,024
  -------------------------------------------------------- --------------- -------------
setRowReadRow(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  -------------------------------------------------------- --------------- -------------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                32
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                32
  -------------------------------------------------------- --------------- -------------
```

커널은 총 1024개의 스레드로 실행되기 때문에 커널에서 실행되는 warp의 수는 총 32개이다. 이 결과를 통해 `setRowReadRow` 커널에서는 bank conflict가 전혀 발생하지 않았기 때문에 각 warp에서 각각 1번의 shared memory load/store transaction이 발생해서 총 load 32번, store 32번의 shared memory transaction이 수행된 것으로 측정되었다.

반면 `setColReadCol` 커널에서는 32-way bank conflict가 발생했고, 이로 인해 각 warp에서는 32번의 순차적인 shared memory transaction이 발생하게 된다. 따라서, 32 warps x 32 shared memory load/store transaction = 1024번의 shared memory load/store transaction이 수행되었다고 측정된 것이다.

## Writing Row-Major and Reading Column-Major

이번에는 row-major order로 shared memory write를 수행하고, column-major order로 shared memory read를 수행하는 커널을 구현해보자. 먼저 row-major order로 shared memory에 global thread index를 write하는 것은 이전 예제와 동일하다.
```c++
tile[threadIdx.y][threadIdx.x] = idx;
```

Column-major order로 shared memory의 값을 global memory에 할당하는 것은 아래와 같이 구현할 수 있다.
```c++
out[idx] = tile[threadIdx.x][threadIdx.y];
```

아래 그림은 bank가 다섯 개라고 가정한 shared memory에서 위의 두 memory operation을 시각화한 것이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbIIxf5%2Fbtr02zaNYdu%2F0hty4EQk0dWn6cKibvntfK%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

전체 커널 구현은 다음과 같다.

```c++
__global__
void setRowReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```

Store operation에서는 row-major order로 액세스하기 때문애 bank conflict가 발생하지 않지만, load operation에서는 column-major order로 액세스하기 때문에 32-way bank conflict가 발생하게 된다. 이는 `nsight compute`를 통해 shared memory store/load transaction을 측정해보면 쉽게 알 수 있다.

```
setRowReadCol(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- -------------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                              1,024
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                 32
  --------------------------------------------------------- --------------- -------------
```

## Dynamic Shared Memory

동적으로 shared memory를 할당하면서 위에서 구현한 `setRowReadCol`과 동일한 동작을 수행하는 커널을 구현할 수도 있다. 동적으로 선언된 shared memory는 반드시 unsized 1D array로 선언되어야 한다. 따라서, 2D thread index 기반으로 memory access index를 계산해야 한다. 여기서도 row-major order write와 column-major order read가 수행되므로, 이를 위해 2개의 인덱스를 계산해야 한다.

- `row_idx` : 1D row-major memory offset
- `col_idx` : 1D column-major memory offset

계산된 `row_idx`를 사용하여 row-major order의 shared memory write를 다음과 같이 작성할 수 있고,
```c++
tile[row_idx] = row_idx;
```

계산된 `col_idx`를 사용하여 column-major order의 shared memory read를 아래와 같이 작성할 수 있다.
```c++
out[row_idx] = tile[col_idx];
```

`out`은 스레드 블록 내에서 row-major order로 정렬되는 global memory 이므로 `row_idx` 위치에 값을 write 한다. 전체 커널 코드는 다음과 같다.
```c++
__global__
void setRowReadColDyn(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int col_idx = blockDim.y * threadIdx.x + threadIdx.y;

    // shared memory store operation
    tile[row_idx] = row_idx;
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}
```

동적으로 shared memory가 선언되기 때문에, 커널을 실행할 때, 이 커널이 사용할 shared memory의 크기를 execution configuration의 세 번째 인자를 통해 지정해주어야 한다. 커널에서 사용하는 shared memory 배열의 크기가 `32x32`의 `int` 타입이므로 다음과 같이 커널을 실행하면 된다.
```c++
setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_out);
```

구현한 `setRowReadColDyn` 커널의 shared memory transaction의 수를 측정해보면 `setRowReadCol`과 동일하게 측정된다.
```
setRowReadColDyn(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- -------------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                              1,024
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                 32
  --------------------------------------------------------- --------------- -------------
```

## Padding Statically Declared Shared Memory

[Shared Memory: Memory Padding](/cuda/study/12_shared_memory.md#memory-padding)에서 **memory padding**에 대해서 알아봤고, 이는 bank conflict를 피하는 방법 중 하나이다. 정적으로 할당된 shared memory에 padding을 적용하는 것은 간단한데, 단순히 innermost 차원에 원하는 크기만큼의 padding을 추가해주면 된다.
```c++
__shared__ int tile[BDIMY][BDIMX + 1];
```

커널 구현은 `setRowReadCol`과 유사하며, 정적으로 할당된 shared memory의 innermost 차원의 크기만 추가될 뿐이다. 결과적으로 각 행마다 하나의 요소만큼 padding이 추가되므로, 같은 열에 있던 요소들은 서로 다른 bank에 속하게 된다. 따라서, column-major order로 수행되는 read 연산에서 발생하던 bank conflict가 사라지게 된다.

전체 커널 구현은 다음과 같다 (매크로를 통해 `PADDING` 값을 1로 지정).
```c++
__global__
void setRowReadColPad(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + PADDING];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```

`nsight compute`를 통해 shared memory store/load transaction을 측정해보면, store와 load에서 모두 bank conflict가 발생하지 않아 둘 다 32로 측정된 것을 확인할 수 있다.
```
setRowReadColPad(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- -----------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                               32
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                               32
  --------------------------------------------------------- --------------- -----------
```

## Padding Dynamically Declared Shared Memory

동적으로 할당된 shared memory에 padding을 추가하는 것은 조금 복잡하다. 2D thread index를 1D memory index로 변환할 때, 반드시 하나의 padding memory space를 스킵해야 한다. 따라서, row-major index와 column-major index를 다음과 같이 계산한다.

```c++
unsigned int row_idx = (blockDim.x + 1) * threadIdx.y + threadIdx.x;
unsigned int col_idx = (blockDim.x + 1) * threadIdx.x + threadIdx.y;
```

아래 그림은 shared memory가 5개의 bank만 있을 때, 하나의 padding이 추가된 경우에서 memory index 계산이 어떻게 되는지 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbpHXC8%2Fbtr01UlMTni%2FqCOWJqkSVyJYpc4hR1q7O1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

전체 커널 구현은 다음과 같다.
```c++
__global__
void setRowReadColDynPad(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = (blockDim.x + 1) * threadIdx.y + threadIdx.x;
    unsigned int col_idx = (blockDim.x + 1) * threadIdx.x + threadIdx.y;
    unsigned int g_idx = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory store operation
    tile[row_idx] = g_idx;
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}
```

동적으로 shared memory를 할당하므로, 커널을 실행할 때 execution configuration을 통해 커널이 사용할 shared memory의 크기를 다음과 같이 지정해주어야 한다.
```c++
setRowReadColDynPad<<<grid, block, (BDIMX + 1) * BDIMY * sizeof(int)>>>(d_out);
```

이 커널에서의 shared memory load/store transaction은 `setRowReadColPad` 커널의 결과와 동일하게 측정된다.
```
setRowReadColDynPad(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- -----------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                               32
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                               32
  --------------------------------------------------------- --------------- -----------
```

## Comparing the Performance of the Square Shared Memory Kernels

지금까지 구현한 모든 square shared memory 커널의 성능은 `nsight system`을 통해 아래와 같이 측정된다 (RTX3080 기준).

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)     GridXYZ         BlockXYZ                Name           
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------  --------------  --------------------------
     23.5            3,648          1   3,648.0   3,648.0     3,648     3,648          0.0     1    1    1    32   32    1  setColReadCol(int *)      
     17.1            2,656          1   2,656.0   2,656.0     2,656     2,656          0.0     1    1    1    32   32    1  setRowReadColDyn(int *)   
     16.9            2,624          1   2,624.0   2,624.0     2,624     2,624          0.0     1    1    1    32   32    1  setRowReadCol(int *)      
     15.6            2,432          1   2,432.0   2,432.0     2,432     2,432          0.0     1    1    1    32   32    1  setRowReadRow(int *)      
     13.6            2,112          1   2,112.0   2,112.0     2,112     2,112          0.0     1    1    1    32   32    1  setRowReadColDynPad(int *)
     13.4            2,080          1   2,080.0   2,080.0     2,080     2,080          0.0     1    1    1    32   32    1  setRowReadColPad(int *)  
```

위 결과를 살펴보면, shared memory padding을 추가해 bank conflict를 모두 제거한 `setRowReadColPad`와 `setRowReadColDynPad` 커널의 성능이 가장 높게 측정된다.

<br>

# Rectangular Shared Memory

이번에는 정사각이 아닌 조금 더 일반적인 2D shared memory 레이아웃인 rectangular shared memory에 대해서 살펴본다. 여기서 shared memory의 행과 열의 수는 동일하지 않다.

정사각 배열의 경우와 달리, 직사각 배열을 참조할 때, 단순히 스레드 `x`, `y` 좌표를 바꾸는 것만으로는 transpose operation을 수행할 수 없다. 정사각 배열에서 구현한 커널을 직사각 배열에 적용하면 memory access violation이 발생한다. 따라서, 이전 섹션에서 square shared memory에 대해 구현했던 모든 커널들을 새로 구현해야 한다. 사실, 완전히 새로 구현하는 것은 아니고 access index만 새로 계산해주면 된다.

> 전체 코드는 [smem_rectangle.cu](/cuda/code/shared_memory_access/smem_rectangle.cu)을 참조

[Squared Shared Memory](#square-shared-memory)에서 구현한 것과 유사하도록 구현하기 위해서 이번에는 각 행은 32개의 요소로 구성되고 각 열은 16개의 요소루 구성되는 rectangular shared memory array를 사용하여 커널을 구현한다. 각 차원의 크기는 아래의 매크로를 통해 정의된다.
```c++
#define BDIMX 32
#define BDIMY 16
```

그리고 각 커널(정적으로 선언된 shared memory를 사용하는 경우)에서는 shared memory tile이 아래와 같이 정적으로 선언된다.
```c++
__shared__ int tile[BDIMY][BDIMX];
```

커널은 하나의 스레드 블록으로 구성되고, rectangular shared memory array와 크기가 동일한 2D block으로 실행된다.
```c++
dim3 block(BDIMX, BDIMY);
dim3 grid(1, 1);
```

## Accessing Row-Major vs. Accessing Column-Major

먼저 square shared memory array에 대해 구현했던 `setRowReadRow`와 `setColReadCol` 커널을 살펴보자.
```c++
__global__ void setRowReadRow(int* out);
__global__ void setColReadCol(int* out);
```

각 커널에서 rectangular shared memory array를 선언할 때, 각 차원의 크기에 주의해야 한다. `setRwoReadRow` 커널에서는 shared memory array의 innermost 차원의 크기는 2D 스레드 블록의 innermost 차원과 동일한 크기로 지정되어야 한다.
```c++
__shared__ int tile[BDIMY][BDIMX];
```

반면 `setColReadCol` 커널에서는 shared memory array의 innermost 차원의 크기가 2D 스레드 블록의 outermost 차원과 동일한 크기로 지정되어야 한다.
```c++
__shared__ int tile[BDIMX][BDIMY];
```

결과적으로 rectangular shared memory에 대한 두 커널의 구현은 square shared memory일 때와 동일하다. 이렇게 구현한 커널들에 대해 `nsight compute`로 shared memory load/store transaction 수를 측정해보면 아래와 같이 측정된다.
```
setColReadCol(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           256
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           256
  --------------------------------------------------------- --------------- --------
setRowReadRow(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                            16
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                            16
  --------------------------------------------------------- --------------- --------
```

`setRowReadRow` 커널은 총 512개의 스레드로 실행된다. 따라서, 총 16개의 warp로 구성되며 각 warp에서 각각 한 번의 shared memory load/store transaction으로 모든 메모리 요청을 처리할 수 있다. 따라서, 각 shared memory load/store transaction의 수가 16으로 측정된다.

반면 `setColReadCol` 커널에서는 각 warp에서 두 개의 스레드 당 동일한 bank의 다른 메모리 주소에 액세스하게 된다. 결과적으로 16-way bank conflict가 발생하게 되고, 16번의 shared memory transaction이 순차적으로 발생한다. 총 16개의 warp가 스레드 블록 내에 존재하기 때문에 load와 store 각각 `16 warps x 16 shared memory transaction = 256`으로 측정되는 것이다.

## Writing Row-Major and Reading Column-Major

Square shared memory에서 구현했던 `setRowReadCol` 커널을 rectangular shared memory에 대해서 구현한다. 실제 행렬 전치를 구현할 때 shared memory를 사용하는 이 커널을 적용하게 되면 성능을 극대화할 수 있다.

2D shared memory tile은 다음과 같이 선언된다.
```c++
__shared__ int tile[BDIMY][BDIMX];
```

이 커널에는 다음의 3가지 memory operation이 수행된다.
- Write to a shared memory row with each warp to avoid bank conflicts
- Read from a shared memory column with each warp to perform a matrix transpose
- Write to a global memory row from each warp with coalesced access

Rectangular shared memory array에서는 먼저 현재 스레드의 2D thread index로부터 1D global thread ID를 다음과 같이 계산한다.
```c++
unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;
```

이는 row-major order이므로 global memory access는 병합된다. 출력은 행과 열이 전치되어야 하므로 transpose matrix에 대한 새로운 좌표를 계산해주어야 한다.
```c++
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
```

그리고 global thread ID 값을 2D shared memory tile에 다음과 같이 저장한다 (shared memory store operation).
```c++
tile[threadIdx.y][threadIdx.x] = idx;
```

각 warp에서는 row-major write를 수행하므로, 이 시점에서는 bank conflict가 발생하지 않는다.

다음으로, 저장한 shared memory data를 읽고, 계산한 `irow`와 `icol` 좌표를 사용하여 전치시켜 global memory에 write한다. 각 warp는 shared memory tile의 한 column을 읽게 된다.
```c++
out[idx] = tile[icol][irow];
```

전체 커널 구현은 다음과 같다.
```c++
__global__
void setRowReadCol(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[icol][irow];
}
```

이 커널의 shared memory transaction을 측정해보면, 다음과 같이 측정된다.
```
setRowReadCol(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           256
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                            16
  --------------------------------------------------------- --------------- --------
```

Shared memory에 global thread ID 값을 저장할 때는 row-major order 이므로, warp 내 모든 스레드들이 서로 다른 bank에 있는 메모리를 요청한다. 하지만, shared memory의 값을 읽을 때는 column-major order 이므로, 조금 전에 살펴본 `setColReadCol` 커널에서와 동일하게 bank conflict가 발생하여 더 많은 shared memory transaction이 수행된다.

## Dynamically Declared Shared Memory

동적으로 shared memory를 선언할 수 있지만, 1차원 배열으로만 선언할 수 있으므로 2D thread 좌표로부터 1D shared memory index를 계산해주어야 한다. Row-major order index는 이미 구해두었으므로, 이전에 계산한 `irow`와 `icol`로부터 column-major order index만 새롭게 계산해주면 된다.
```c++
unsigned int col_idx = blockDim.x * icol + irow;
```

전체 커널 구현은 다음과 같다.
```c++
__global__
void setRowReadColDyn(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx];
}
```

동적으로 shared memory를 할당하므로, 커널을 실행할 때 execution configuration을 통해 이 커널에 할당되는 shared memory 크기를 지정해주어야 한다.
```c++
setRowReadCol<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_out);
```

이 커널의 shared memory transaction 수는 `setRowReadCol` 커널과 동일하다.
```
setRowReadColDyn(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           256
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                            16
  --------------------------------------------------------- --------------- --------
```

## Padding Statically Declared Shared Memory

Square shared memory 커널에서 적용한 것과 동일하게 memory padding을 추가하여 bank conflict를 해결할 수 있다. 하지만, padding 요소가 얼마나 필요한지는 계산해야 한다.

우선 전체 커널 구현부터 살펴보자. Padding 크기는 매크로 `PADDING`을 통해 지정한다.
```c++
__global__
void setRowReadColPad(int* out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + PADDING];
    
    // mapping from thread index to global memory index
    unsigned int idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads(); // wait for all threads to complete

    // shared memory load operation
    out[idx] = tile[icol][irow];
}
```

이제 `PADDING`의 값을 고민해야 하는데, 먼저, square shared memory array 때와 동일하게 1로 지정하고, `nsight compute`를 통해 shared memory transaction을 측정하면 다음과 같은 결과를 얻을 수 있다.
```
setRowReadColPad(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                            32
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                            16
  --------------------------------------------------------- --------------- --------
```
Shared memory load operation에서 bank conflict가 완전히 사라지지 않았다는 것을 알 수 있다. 이는 padding을 추가하기 전을 기준으로 wrap 내 스레드들이 절반씩 나누어 2개의 bank에 액세스하기 때문이다.

예를 들면, 처음 16개의 스레드들은 bank 0에 있는 서로 다른 메모리 주소를 요청하고, 나머지 16개의 스레드들은 bank 16에 있는 서로 다른 메모리 주소를 요청한다. 따라서, padding 크기를 1로 지정하면, 처음 16개의 스레드들이 요청하는 메모리 주소가 서로 다른 bank에 위치하게 되지만, thread 15(16번째 스레드)가 요청하는 메모리 주소가 속한 bank(15)는 나머지 16개의 스레드 중 첫 번째 스레드(thread 16)이 속한 bank와 일치하게 된다.

따라서, 지금과 같이 `BDIMX`가 32이고, `BDIMY`가 16인 경우에는 padding의 크기를 2로 지정해주어야 bank conflict를 완전히 제거할 수 있게 된다. `PADDING`의 크기를 2로 지정한 뒤, 측정한 shared memory transaction은 다음과 같다.
```
setRowReadColPad(int *), Context 1, Stream 7
  Section: Command line profiler metrics
  --------------------------------------------------------- --------------- --------
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                            16
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                            16
  --------------------------------------------------------- --------------- --------
```

## Padding Dynamically Declared Shared Memory

Square shared memory에서와 마찬가지로 동적으로 shared memory를 선언하는 경우, memory padding을 추가하는 것이 조금 더 복잡하다. 1D shared memory index를 계산해야 하는데 padding이 추가된 shared memory와 global memory는 서로 다른 크기가 되므로, 몇몇 인덱스를 다시 계산해주어야 한다. 계산 방법은 square shared memory와 유사하므로 따로 설명하지는 않는다.

전체 커널 구현은 다음과 같다.

```c++
__global__
void setRowReadColDynPad(int* out)
{
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int g_idx = blockDim.x * threadIdx.y + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;
    
    unsigned int row_idx = (blockDim.x + PADDING) * threadIdx.y + threadIdx.x;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * (blockDim.x + PADDING) + irow;

    // shared memory store operation
    tile[row_idx] = g_idx;
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}
```

이 커널의 shared memory transaction 수는 `setRowReadColPad`와 동일하며, bank conflict가 완전히 제거된다.

## Comparing the Performance of the Rectangular Shared Memory Kernels

지금까지 구현한 rectangular shared memory에 대한 모든 커널의 성능을 `nsight system`으로 측정한 결과는 다음과 같다.

```
$ nsys profile --stats=true ./smem_rectangle
...
Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)     GridXYZ         BlockXYZ                Name           
--------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------  --------------  --------------------------
    18.8            2,560          1   2,560.0   2,560.0     2,560     2,560          0.0     1    1    1    32   16    1  setColReadCol(int *)      
    18.1            2,464          1   2,464.0   2,464.0     2,464     2,464          0.0     1    1    1    32   16    1  setRowReadRow(int *)      
    16.2            2,208          1   2,208.0   2,208.0     2,208     2,208          0.0     1    1    1    32   16    1  setRowReadColDyn(int *)   
    16.0            2,176          1   2,176.0   2,176.0     2,176     2,176          0.0     1    1    1    32   16    1  setRowReadCol(int *)      
    15.5            2,112          1   2,112.0   2,112.0     2,112     2,112          0.0     1    1    1    32   16    1  setRowReadColDynPad(int *)
    15.3            2,080          1   2,080.0   2,080.0     2,080     2,080          0.0     1    1    1    32   16    1  setRowReadColPad(int *)
```

Memory padding을 추가하여 bank conflict를 제거한 커널에서 조금 더 좋은 성능을 보여주고 있다.

> 동적으로 shared memory를 할당 및 선언하는 경우, 약간의 오버헤드가 있다고 언급하고 있다. 일단 성능 측정 결과에서는 오버헤드의 증거를 명확히 찾을 수는 없었다. 개인적으로 동적이든 정적이든 SM에 active warp가 있을 때 할당하는 것은 동일하므로 거의 유사한 오버헤드가 있을 것 같다고 생각된다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Memory Optimizations - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [NVIDIA CUDA Documentation: Compute Capability 5.x - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)