# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [Indexing Matrices with Blocks and Threads](#indexing-matrices-with-blocks-and-threads)
- [Matrices with a 2D Grid and 2D Blocks](#matrices-with-a-2d-grid-and-2d-blocks)
- [1D Grid and 1D Blocks](#1d-grid-and-1d-blocks)
- [2D Grid and 1D Blocks](#2d-grid-and-1d-blocks)
- [References](#references)

<br>

# Intro

[CUDA Programming Model](/cuda/study/02_cuda_programming_model.md)에서 벡터 덧셈 예제를 통해 그리드와 블록 사이즈를 사용하여 스레드를 어떻게 조직화하는지 살펴볼 수 있다. 스레드를 어떻게 조직화하느냐에 따라 커널의 성능에 큰 영향을 미칠 수 있는데, 이전의 벡터 덧셈 예제([vector_add.cu](/cuda/code/vector_add/vector_add.cu)에서 블록의 크기를 조절하여 최적의 성능을 찾을 수 있다. 그리드의 크기는 사실 데이터의 수와 블록 크기를 통해 계산되므로 블록 크기에 의해 결정된다고 볼 수 있다.

> 이번 포스팅을 준비하면서 그리드와 블록 크기에 따른 성능을 비교했는데, naive한 구현에서는 그리드와 블록 크기에 따른 성능 차이는 거의 없었다 (구현에 따른 차이는 제외).
>
> `RTX 3080`으로 테스트를 수행했는데, 예전의 오래된 GPU들보다 넉넉한 CUDA Cores, SMs를 가지고 있기 때문에 그런 것이 아닌가 추측된다. 아직 다루지는 않았지만, 메모리 액세스 패턴이나 다른 최적화 기법들이 들어가게 된다면 그리드나 블록의 크기가 꽤 영향을 미칠 것으로 보인다.
>
> 이번 포스팅에서는 그리드와 블록의 차원을 이런식으로 설정할 수 있다는 것에만 주목하면 될 것 같다.

코드 원본에서는 블록 당 256개의 스레드를 가지고 있다. 이때, $2^{20}$ 개의 요소에 대해 실행해보면 아래와 같은 결과를 얻을 수 있다.
```
[Vector addition of 1048576 elements on GPU]
> Copy input data from the host memory to the CUDA device
> CUDA kernel launch with 4096 blocks of 256 threads
> Copy output data from the CUDA device to the host memory
> Verifying vector addition...
> Test PASSED
Performance = 37.75 GFlop/s, Time = 0.028 msec, Size = 1048576 ops,  WorkgroundSize = 256 threads/block
Done
```

블록 당 스레드의 수를 512개로 변경해서 테스트하면, 0.023ms까지 성능이 증가한 것을 관찰했다.
```
[Vector addition of 1048576 elements on GPU]
> Copy input data from the host memory to the CUDA device
> CUDA kernel launch with 2048 blocks of 512 threads
> Copy output data from the CUDA device to the host memory
> Verifying vector addition...
> Test PASSED
Performance = 44.77 GFlop/s, Time = 0.023 msec, Size = 1048576 ops,  WorkgroundSize = 512 threads/block
Done
```
사실 이정도 차이는 측정 오차라고 봐도 무방할 정도긴 하지만 아마 수백번 돌려서 평균을 측정해보면, 조금이라도 성능이 좋아진 것을 확인할 수도 있을 것이다.

> 복잡한 연산이 아니고, 커널이 단순하여 블록 사이즈에 큰 영향을 받지 않는 것이 아닌가 추측한다.

이번 포스팅에서는 행렬 덧셈 예제를 통해 조금 더 자세히 살펴본다. 행렬 연산을 CUDA로 구현한다고 하면, 자연스럽게 데이터의 레이아웃은 2차원 블록의 2차원 그리드라고 생각할 수 있다 (2D grid with 2D blocks). 아래 내용들을 살펴보면 naive한 접근 방법이 최적의 성능이 아니라는 것을 볼 수 있다. 그리고 그리드와 블록 사이즈를 아래의 레이아웃들로 구성하면 행렬 덧셈의 성능이 어떻게 변하는지 살펴본다.

- 2D grid with 2D blocks
- 1D grid with 1D blocks
- 2D grid with 1D blocks

<br>

# Indexing Matrices with Blocks and Threads

일반적으로 프로그램에서 행렬은 global memory에 row-major layout으로 선형적으로 저장된다. 아래 그림에서는 8x6 행렬이 일반적으로 저장되는 경우를 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5f3ua%2FbtrYyTW7Wot%2FkZxjI1cZs5kezf6dI2slHK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

잠시 뒤 아래에서 볼 행렬 덧셈 커널에서 하나의 스레드는 행렬 요소 하나를 처리한다 (naive 구현). 각 스레드는 블록과 스레드의 인덱스를 사용하여 자신이 어떤 데이터를 처리할 지 알아낸다.
일반적으로 2차원인 경우, 다루어야할 인덱스의 종류는 아래와 같다.

- thread and block index
- coordinate of a given point in the matrix
- offset in linear global memory

주어진 한 스레드에서 블록 인덱스와 스레드 인덱스를 행렬의 좌표에 매핑하여 처리해야 할 데이터가 global memory에서 몇 번째 위치하는지 offset을 구할 수 있다. 즉, 행렬 좌표를 사용하여 global memory location을 찾을 수 있다.

정리하면, 아래의 순서로 처리해야 할 데이터가 주어진 메모리의 어디에 위치하는지 찾는다.

1. 블록과 스레드 인덱스를 사용하여 행렬의 좌표를 찾는다.

```
ix = threadIdx.x + blockIdx.x * blockIdx.y
iy = threadIdx.y + blockIdx.y * blockIdx.y
```

2. 1에서 찾은 행렬 좌표를 통해 global memory location/index를 구한다.

```
idx = iy * nx + ix
```

아래 그림에서 블록과 스레드 인덱스, 행렬 좌표, 선형 global memory 인덱스의 관계를 보여준다. 초록색 네모 하나가 블록 하나를 의미한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F1JlrK%2FbtrYzqAb1yY%2FC03Lg4S0sKl4jfkau2hVx1%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

조금 더 자세히 이해하고자 하면, 아래 코드에서 `printThreadInfo` 커널을 살펴보면 좋다. 이 커널에서는 각 스레드의 thread index / block index / matrix coordinate / (linaer)global memory offset를 출력해준다.

<details>
<summary>printThreadInfo code 보기</summary>

```c++
#include <stdio.h>
#include <cuda.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

__global__
void printThreadIndex(int* A, int const nx, int const ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d)  block_id (%d,%d)  coordinate (%d,%d)  global_index %2d  val %2d\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char** argv)
{
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(int);

    // malloc host memory
    int* h_A;
    h_A = static_cast<int*>(malloc(n_bytes));

    // init host matrix with integer
    for (int i = 0; i < nxy; i++) {
        h_A[i] = i;
    }

    // malloc device memory
    int* d_A;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, n_bytes));

    // transfer data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));

    // set up execution configuration (grid and block size)
    dim3 block(4, 2);
    dim3 grid((nx + block.x -1) / block.x, (ny + block.y - 1) / block.y);

    // launch the kernel
    printThreadIndex<<<grid,  block>>>(d_A, nx, ny);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // free host and device memory
    CUDA_ERROR_CHECK(cudaFree(d_A));
    free(h_A);

    // reset device
    cudaDeviceReset();

    return 0;
}
```

</details>


위 코드는 8x6 크기의 행렬에 대해 실행하며 아래와 같이 출력된다.
```
thread_id (0,0)  block_id (1,2)  coordinate (4,4)  global_index 36  val 36
thread_id (1,0)  block_id (1,2)  coordinate (5,4)  global_index 37  val 37
thread_id (2,0)  block_id (1,2)  coordinate (6,4)  global_index 38  val 38
thread_id (3,0)  block_id (1,2)  coordinate (7,4)  global_index 39  val 39
thread_id (0,1)  block_id (1,2)  coordinate (4,5)  global_index 44  val 44
thread_id (1,1)  block_id (1,2)  coordinate (5,5)  global_index 45  val 45
thread_id (2,1)  block_id (1,2)  coordinate (6,5)  global_index 46  val 46
thread_id (3,1)  block_id (1,2)  coordinate (7,5)  global_index 47  val 47
thread_id (0,0)  block_id (0,0)  coordinate (0,0)  global_index  0  val  0
thread_id (1,0)  block_id (0,0)  coordinate (1,0)  global_index  1  val  1
thread_id (2,0)  block_id (0,0)  coordinate (2,0)  global_index  2  val  2
thread_id (3,0)  block_id (0,0)  coordinate (3,0)  global_index  3  val  3
thread_id (0,1)  block_id (0,0)  coordinate (0,1)  global_index  8  val  8
thread_id (1,1)  block_id (0,0)  coordinate (1,1)  global_index  9  val  9
thread_id (2,1)  block_id (0,0)  coordinate (2,1)  global_index 10  val 10
thread_id (3,1)  block_id (0,0)  coordinate (3,1)  global_index 11  val 11
...
```

이렇게 출력되는 스레드를 시각적으로 나타내면 아래처럼 나타낼 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGS7qX%2FbtrYz8y3xvS%2FYudUfRaId83ZS74CC8q49K%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

<br>

# Matrices with a 2D Grid and 2D Blocks

> 전체 코드는 [matrix_add.cu](/cuda/code/matrix_add/matrix_add.cu) 를 참조

먼저 살펴볼 내용은 2D grid with 2D blocks를 사용하는 행렬 덧셈 커널이다. 2D 스레드 블록을 사용하여 행렬을 더하는 커널은 아래와 같이 작성할 수 있다.
```c++
__global__
void sumMatrixOnGPU2D(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}
```

이 커널에서의 핵심은 각 스레드들이 global linear memory 인덱스에 어떻게 매핑되냐는 것인데, 아래 그림에서 이에 대해 자세히 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FPm7Oq%2FbtrYvm6q584%2FHzkuakcdlTSQqCwI9n1ul0%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

[matrix_add.cu](/cuda/code/matrix_add/matrix_add.cu) 를 컴파일하고, 실행하면 아래와 같은 출력 결과를 얻을 수 있다. 아래 출력은 $2^{14} \times 2^{14}$ 행렬에 대해 스레드 블록의 크기를 (16,16)으로 지정하여 실행시킨 결과이다.

```
> Matrix size: 16384 x 16384
> sumMatrixOnHost  Elapsted Time: 353.958 msec
> sumMatrixOnGPU2D<<<(1024,1024), (16,16)>>> (Average)Elapsted Time: 4.625 msec
> Verifying vector addition...
> Test PASSED
```

블록의 크기를 (32,32), (32,16), (16,32), (16,16)으로 각각 지정했을 때의 결과를 요약하면 아래와 같다.

|Block Size|Average Elapsed Time(100 times)|Grid Size|
|-|:-:|-|
|(32,32)|4.677 ms|(512,512)|
|(32,16)|4.622 ms|(512,1024)|
|(16,32)|4.635 ms|(1024,512)|
|(16,16)|4.625 ms|(1024,1024)|

차이가 0.01ms 단위로 발생한다. 사실상 측정 오차에 가깝다고 볼 수 있을 것 같다. 하지만, 한 가지 분명한 것은 블록의 크기가 크다고 항상 빠르지 않다는 것이다.

<br>

# 1D Grid and 1D Blocks

이번에는 1D grid with 1D blocks를 사용하는 커널에 대해 테스트를 진행한다. 이 커널에서는 아래 그림과 같이 스레드와 데이터 간의 매핑이 이루어진다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fk4HPX%2FbtrYAmw9fp1%2F8dU3p0kHkaC5kXetPVtgi0%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

즉, 하나의 스레드가 하나의 column(ny개의 요소)을 담당하게 된다. 커널 구현은 다음과 같다.
```c++
__global__
void sumMatrixOnGPU1D(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx) {
        int idx;
        for (int iy = 0; iy < ny; iy++) {
            idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}
```

1차원 블록을 사용하므로 2D grid with 2D blocks와 데이터 매핑 방법이 다르다. 여기서는 오직 `threadIdx.x`만을 사용하고, 각 스레드는 이제 하나의 데이터가 아닌 `ny`개의 요소들을 처리한다.

블록의 크기를 (256,1)로 지정하고, 실행한 결과는 아래와 같다.
```
> Matrix size: 16384 x 16384
> sumMatrixOnHost  Elapsted Time: 345.341 msec
> sumMatrixOnGPU1D<<<(64,1), (256,1)>>> (Average)Elapsted Time: 7.025 msec
> Verifying vector addition...
> Test PASSED
```

블록이 크기를 각각 (1024,1), (512,1), (256,1), (128,1)로 지정했을 때의 결과

|Block Size|Average Elapsed Time(100 times)|Grid Size|
|-|:-:|-|
|(1024,1)|7.011 ms|(16,1)|
|(512,1)|7.026 ms|(32,1)|
|(256,1)|7.025 ms|(64,1)|
|(128,1)|7.047 ms|(128,1)|

결과를 보면, 2D grid with 2D blocks 보다 전체적으로 느려졌다는 것을 볼 수 있다. 이는 하나의 스레드가 담당하는 데이터가 많아지면서, 병렬 처리 능력이 떨어졌다라고 해석할 수 있다.

<br>

# 2D Grid and 1D Blocks

마지막으로 살펴볼 방법은 1D block들로 구성된 2D gird를 사용하는 커널이다. 이 접근 방법으로 구현된 커널에서는 첫 번째 방법(2D grid with 2D blocks)와 같이 하나의 스레드가 하나의 데이터에 매핑된다. 그림으로 나타내면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHNXI9%2FbtrYzqtsCFc%2FPaVxzY3bATPTkdgUvrOH00%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

커널 구현은 아래와 같다.

```c++
__global__
void sumMatrixOnGPUMix(float const* A, float const* B, float* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}
```

아래는 블록의 크기를 (256,1)로 지정했을 때의 결과이다.
```
> Matrix size: 16384 x 16384
> sumMatrixOnHost  Elapsted Time: 350.448 msec
> sumMatrixOnGPUMix<<<(64,16384), (256,1)>>> (Average)Elapsted Time: 4.619 msec
> Verifying vector addition...
> Test PASSED
```

결과를 보면, 첫 번째 방법(2D grid with 2D blocks)과 성능이 비슷한 것을 볼 수 있다. 이는 하나의 스레드가 하나의 요소를 처리하는 것이 동일하여 결과적으로 동일한 병렬 처리를 수행하기 때문이라고 볼 수 있을 것 같다. 두 커널과 같이 naive한 구현에서는 그리드와 블록의 레이아웃을 바꿔도 성능의 차이는 크지 않다고 볼 수 있을 것 같다.

블록의 크기를 각각 (1024,1), (512,1), (256,1), (128,1)로 지정했을 때의 결과는 다음과 같다.

|Block Size|Average Elapsed Time(100 times)|Grid Size|
|-|:-:|-|
|(1024,1)|4.623 ms|(16,16384)|
|(512,1)|4.608 ms|(32,16384)|
|(256,1)|4.619 ms|(64,16384)|
|(128,1)|4.612 ms|(128,16384)|

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher