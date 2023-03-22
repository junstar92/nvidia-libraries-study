# Table of Contents

- [Table of Contents](#table-of-contents)
- [Shared Memory](#shared-memory)
- [Example: Matrix Multiplication](#example-matrix-multiplication)
- [References](#references)

<br>

# Shared Memory

Shared memory는 `__shared__` speicifier를 사용하여 할당된다.

Shared memory는 global memory보다 훨씬 더 빠르다. 일반적으로 CUDA 블록에서 global memory access를 최소화하기 위해 shared memory를 `scratchpad memory`(or software managed cache)로 사용할 수 있다.

행렬 곱셈 예제를 통해서 shared memory를 사용하는 방법과 shared memory를 사용했을 때 성능이 얼마나 향상되는지 살펴보자.

# Example: Matrix Multiplication

행렬곱을 host 코드로 구현하면 아래와 같이 구현할 수 있다.

```c++
void matmulHost(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            float sum = 0.f;
            for (int i = 0; i < k; i++) {
                sum += A[k * r + i] * B[n * i + c];
            }
            C[n * r + c] = sum;
        }
    }
}
```

단순히 행렬 곱셈 공식을 코드로 표현했고, 3중 for문으로 구현된다. 따라서, 입력 행렬의 크기가 크다면 시간이 매우 오래 걸린다.

결과 행렬인 `C` 행렬의 요소 하나하나를 독립적으로 병렬로 계산할 수 있다. 따라서, 이를 naive하게 CUDA 커널 함수로 구현하면 다음과 같다.
```c++
__global__
void matmulNaive(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.f;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += A[k * row + i] * B[n * i + col];
        }
        C[n * row + col] = sum;
    }
}
```

위 커널에서 각 스레드는 `C` 행렬의 요소 하나를 계산한다. 따라서, 하나의 스레드에서는 `k`차원만큼 for문을 반복하면서 `A`와 `B` 행렬의 요소곱을 수행한다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png" height=500px style="display: block; margin: 0 auto"/>

그림으로 표현하면 위와 같다. 가만히 살펴보면, 각 스레드가 하나의 요소를 계산할 때 액세스하는 `A`와 `B` 행렬의 요소가 중복된다는 것을 볼 수 있다. 즉, 불필요한 global memory access가 발생하게 되며, global memory에 액세스하는 latency는 비교적 큰 편이므로 최대한 중복되는 액세스를 제거하는 것이 좋다.

이러한 문제는 shared memory를 사용하여 해결할 수 있다. Shared memory는 on-chip memory이므로 global memory보다 latency가 짧다. 따라서, 중복되는 global memory 액세스를 shared memory 액세스로 바꾸면 조금 더 빠르게 연산을 수행할 수 있다.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png" height=500px style="display: block; margin: 0 auto"/>

원리는 간단하다. 하나의 스레드 블록, 즉, 결과인 `C` 행렬의 일부 요소에 대해 계산할 때 중복해서 사용되는 요소들을 shared memory에 먼저 저장한다. 그리고, 저장한 shared memory를 통해 요소곱을 수행한다. 따라서, 더 이상 global memory가 아닌 shared memory를 반복적으로 액세스하게 되며, shared memory가 latency가 짧기 때문에 더 빠른 실행 속도를 얻을 수 있는 것이다. 이와 같은 방법을 **tiling** 기법이라고 부른다.

이를 아래의 코드로 구현할 수 있다.

```c++
template<int BLOCK_SIZE>
__global__
void matmulSmem(float const* A, float const* B, float* C, int const m, int const k, int const n)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int n_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int c_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    unsigned int c_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int a_row, a_col, a_idx, b_row, b_col, b_idx;

    float sum = 0.f;
    // loop over all the sub-matrices of A and B
    // to compute the block sub-matrix
    for (unsigned int block = 0; block < n_blocks; block++) {
        __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

        // calculate row, column, data index
        a_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        a_col = BLOCK_SIZE * block + threadIdx.x;
        a_idx = a_row * k + a_col;
        b_row = BLOCK_SIZE * block + threadIdx.y;
        b_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        b_idx = b_row * n + b_col;

        // load the matrices from global memory to shared memory
        Asub[threadIdx.y][threadIdx.x] = (a_row < m && a_col < k) ? A[a_idx] : 0.f;
        Bsub[threadIdx.y][threadIdx.x] = (b_row < k && b_col < n) ? B[b_idx] : 0.f;
        cta.sync(); // synchronize to make sure the matrices are loaded

        // multiply the two matrices
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }
        cta.sync();
    }

    // write the block sub-matrix to global memory
    if (c_row < m && c_col < n) {
        C[c_row * n + c_col] = sum;
    }
}
```

위 코드에서는 템플릿을 사용하여 스레드 블록의 x,y 차원 크기를 지정하고 있다. 이는 static shared memory를 사용하기 위함이며, dynamic shared memory를 사용한다면 굳이 템플릿을 사용하지 않아도 된다 (다른 상수 변수로 `BLOCK_SIZE`를 지정할 수 있다면, 템플릿을 사용하지 않고도 static shared memory를 사용할 수 있다). 위 코드에서는 shared memory 배열을 2차원 배열로 사용하기 위해서 static shared memory를 사용한다.

> 전체 코드는 [matmul.cu](/cuda/code/matmul/matmul.cu)에서 확인할 수 있음

> `matmulSmem` 커널은 `BLOCK_SIZE` 크기로 나누어 떨어지는 행렬 크기뿐만 아니라 다양한 크기의 행렬 곱셈을 지원할 수 있도록 구현되어 있다. 따라서, `BLOCK_SIZE` 크기로 나누어 떨어지지 않더라도 정확한 행렬 곱셈 연산이 가능하다.

[matmul.cu](/cuda/code/matmul/matmul.cu) 코드를 컴파일하고, 실행하면 아래의 결과를 얻을 수 있다.
```
$ ./matmul
> Starting matrix multiplication at device 0: NVIDIA GeForce RTX 3080
> Matrix A  : (1024 x 1024)
> Matrix B  : (1024 x 1024)
> BLOCK_SIZE: 32
matmulHost      : 3065.07 ms
matmulNaive     : 1.3591 ms
matmulSmem      : 0.994368 ms
```
위 결과는 `A(1024x1024)`행렬과 `B(1024x1024)`행렬의 곱셈 결과이다. Host에서의 계산은 약 3초가 걸리고, shared memory를 사용하지 않고 구현한 커널에서는 약 1.36ms가 걸린다. 반면 shared memory를 사용한 경우에는 약 0.36ms 정도 빠른 1ms가 걸리는 것을 확인할 수 있다.

참고로, 아래와 같이 행렬의 각 차원 크기를 `BLOCK_SIZE`에 나누어 떨어지지 않는 값으로 설정해도 정상적으로 계산하는 것을 볼 수 있다.
```
$ ./matmul 228 240 112
> Starting matrix multiplication at device 0: NVIDIA GeForce RTX 3080
> Matrix A  : (228 x 240)
> Matrix B  : (240 x 112)
> BLOCK_SIZE: 32
matmulHost      : 3.40378 ms
matmulNaive     : 0.028288 ms
matmulSmem      : 0.019456 ms
```

<br>

아래 포스팅에서 shared memory에 대해 조금 더 자세히 다루고 있다.

- [Shared Memory](/cuda/study/12_shared_memory.md)
- [Data Layout of Shared Memory](/cuda/study/12-1_data_layout_of_shared_memory.md)
- [Reducing Global Memory Access](/cuda/study/12-2_reducing_global_memory_access.md)
- [Coalescing Global Memory Accesses](/cuda/study/12-3_coalescing_global_memory_accesses.md)

<br>

# References

- [NVIDIA CUDA Documentations: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- CUDA sample code: [matrixMul.cu](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu)