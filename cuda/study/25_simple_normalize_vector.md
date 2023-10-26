# Table of Contents

- [Table of Contents](#table-of-contents)
- [Asynchronous Barrier](#asynchronous-barrier)
- [Normalize Vector by Dot Product](#normalize-vector-by-dot-product)
- [References](#references)

<br>

# Asynchronous Barrier

이번 포스팅에서는 [link](/cuda/doc/01_programming_guide/07-26_asynchronous_barrier.md)에서 다루었던 Asynchronous Barrier에 대해서 다시 살펴보고, CUDA 샘플 코드를 통해서 무엇을 할 수 있는지 살펴보도록 하자.

> Asynchronous Barrier에 대한 전체 내용은 [link](/cuda/doc/01_programming_guide/07-26_asynchronous_barrier.md)를 참조

먼저 CUDA C++에서 제공하는 asynchronous barrier가 없다면 `__syncthreads()` 또는 cooperative groups를 이용한 `group.sync()`를 통해 스레드 블록 내 모든 스레드를 동기화시킬 수 있다.
```c++
#include <cooperative_groups.h>

__global__
void simple_sync(int iter_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iter_count; i++) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

위의 코드는 세 단계(before sync, sync point, after sync)로 구성된다.

`cuda::barrier`를 사용하면 아래와 같이 temporally-split synchronization pattern을 나타낼 수 있다.
```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* code before arrive */
       barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
       compute(data, curr_iter);
       bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/
        /* code after wait */
    }
}
```

`cuda::barrier`를 사용하려면 먼저 `init()` 함수를 통해서 초기화를 해주어야 하는데, `init()` 함수에는 배리어에 도착할 예상 스레드 갯수를 인자로 전달한다. 위 코드에서는 `block.size()`를 전달하였으므로 해당 스레드 블록 내 모든 스레드들이 참여한다는 것을 의미한다. 이는 배리어에 참여하는 스레드가 `bar.wait(std::move(token))`을 호출하여 블로킹이 해제되기 전에 `bar.arrive()`를 호출하는 횟수와 동일하다.

`cuda::barrier`는 네 단계로 구성되는데, 각 단계는 아래와 같다.

- Arrival : 각 스레드가 `bar.arrive()` 호출
- Countdown : 각 스레드가 `bar.arrive()`에 도착할 때마다 카운트다운
- Completion : 마지막 `bar.arrive()`로 인해 카운트다운이 0이되면 `cuda::barrier`는 complete
- Reset : 카운트다운이 0이되면 자동으로 리셋

위 코드에서 `token`은 `bar.arrive()`로부터 리턴되며, barrier의 현재 단계와 연관되어 있다. 즉, `bar.wait(std::move(token))` 호출은 `cuda::barrier`가 현재 단계에 있는 동안 이를 호출한 스레드를 블로킹한다. 만약 카운트다운이 0가 되어서 `cuda::barrier`가 다음 단계로 진행되었다면 `bar.wait(std::move(token))`은 해당 스레드를 블로킹하지 않고 바로 리턴하게 된다. 만약 스레드가 `bar.wait(std::move(token))`에서 블록되어 있는 상태이면서 `cuda::barrier`가 다음 단계로 진행하면, 해당 스레드에서의 블로킹이 해제된다.


# Normalize Vector by Dot Product

CUDA 샘플 코드에서 간단하게 `cuda::barrier`를 사용하는 [SimpleAWBarrier](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleAWBarrier) 예제를 통해 어떻게 사용될 수 있는지 살펴보자.

위 예제에서 풀려는 문제는 주어진 두 벡터의 normalization을 구하는 것이다. 즉, 아래의 식을 계산하는 것이 풀고자하는 문제이다.

$$ \begin{matrix} A' = \frac{A}{A \cdot B} && B' = \frac{B}{A \cdot B} \end{matrix} $$

여기서 $A\cdot B$ 계산을 병렬로 처리해야 할 것이다. 다만, 벡터의 요소의 수가 너무 많다면 주어진 스레드 블록 만으로는 힘들다. 먼저 벡터를 스레드 블록 단위로 나누고, 그리드에는 총 4개의 스레드 블록만 존재하고 벡터의 요소 수는 4개의 스레드 블록의 두 배 크기라고 가정해보자. 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbrvHhX%2FbtszdqLiRw8%2FY9Tfy5scbfQXR4LJ2SuTu0%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

그렇다면 스레드 블록 내 개별 스레드들은 그리드의 차원 크기만큼 iteration을 돌면서 A와 B의 각 요소 당 내적 값을 누적하게 된다. 위 그림에서 초록색으로 표시된 결과가 이에 해당한다. 이 결과는 개별 스레드의 레지스터에 저장된다.

그런 다음 스레드 블록 단위로 reduction을 수행한다. 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Frtww3%2FbtszbOzv7ml%2F3mWpsOTvUGK8SMuRipbHWk%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

한 스레드 블록에서의 reduction은 우선 워프 단위로 수행한다. 그 결과, 각 스레드 블록에서는 스레드 블록 내 워프 갯수로 reduction이 수행된다. 이때, 워프 단위에서 reduction한 결과는 최종적으로 0번째 워프가 취합해야 하므로 shared memory에 저장한다. 모든 워프에서 저장이 완료된 이후에 0번째 워프가 계산해야 됨을 보장하기 위해서 이 지점에 동기화가 필요하다. 샘플 코드는 이 포인트에서 `bar.arrive()`를 호출했다. 이 지점에 도달한 이후에는 사실 0번째 워프만 연산을 계속 진행하기 때문에 `bar.wait()`를 곧바로 호출하게 되는 것을 코드를 통해 확인할 수 있다.

워프 단위로 계산된 결과는 각 스레드 블록 인덱스 위치의 global memory 배열에 저장이 된다. 그런 다음, 첫 번째 스레드 블록만 연산에 참여하여 각 개별 스레드가 global memory 배열 전체를 각 레지스터에 reduction 한 다음, 다시 한 번 이 스레드 블록에서만 워프 단위로 reduction을 수행하여 최종적으로 내적값을 계산하게 된다.

전체 커널 구현은 다음과 같다. 스레드 블록 내에서 워프 단위로 reduction을 수행하는 코드를 device 함수로 별도로 작성했으며, 템플릿 변수를 통해 마지막 워프 단위 reduction에서는 최종 내적값에 `sqrt`를 씌워서 global memory에 다시 저장하게 된다. 그런 다음, 이 결과 값을 각 A, B 벡터의 요소에 나누어 주면 계산이 끝나게 된다.

```c++
template<bool WriteSquareRoot>
__device__
void reduce_block_data(
    cuda::barrier<cuda::thread_scope_block>& barrier,
    cg::thread_block_tile<32>& tile32, double& thread_sum, double* result
)
{
    extern __shared__ double tmp[];

    #pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        thread_sum += tile32.shfl_down(thread_sum, offset);
    }
    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = thread_sum;
    }

    auto token = barrier.arrive();
    barrier.wait(std::move(token));

    // The warp 0 will perform last round of reduction
    if (tile32.meta_group_rank() == 0) {
        double beta = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.;

        #pragma unroll
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            beta += tile32.shfl_down(beta, offset);
        }

        if (tile32.thread_rank() == 0) {
            if (WriteSquareRoot)
                *result = sqrt(beta);
            else
                *result = beta;
        }
    }
}

__global__
void norm_vec_by_dot_product(float* vecA, float* vecB, double* partialResults, int size)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    cg::sync(cta);

    double thread_sum = 0.;
    for (int i = grid.thread_rank(); i < size; i+= grid.size()) {
        thread_sum += (double)(vecA[i] * vecB[i]);
    }

    // Each thread block performs reduction of partial dot products and 
    // writes to global memory
    reduce_block_data<false>(barrier, tile32, thread_sum, &partialResults[blockIdx.x]);

    cg::sync(grid);

    // One block performs the final summation of partial dot products
    // of all the thread blocks and writes the sqrt of final dot product
    if (blockIdx.x == 0) {
        thread_sum = 0.;
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
            thread_sum += partialResults[i];
        }
        reduce_block_data<true>(barrier, tile32, thread_sum, &partialResults[0]);
    }
    cg::sync(grid);

    const double final_value = partialResults[0];

    // Perform normalization of vector A & B
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        vecA[i] = (float)vecA[i] / final_value;
        vecB[i] = (float)vecB[i] / final_value;
    }
}
```

> 전체 코드는 [norm_vec.cu](/cuda/code/normalize_vector/norm_vec.cu)에서 확인할 수 있다.


# References

- [NVIDIA CUDA Documentation: Asynchronous Barrier](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/index.html#asynchronous-barrier)
- [cuda-samples: SimpleAWBarrier](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleAWBarrier)