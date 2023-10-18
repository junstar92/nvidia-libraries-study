# Table of Contents

- [Table of Contents](#table-of-contents)
- [`memcpy_async` API](#memcpy_async-api)
- [Copy and Compute Pattern - Staging Data Through Shared Memory](#copy-and-compute-pattern---staging-data-through-shared-memory)
- [Without `memcpy_async`](#without-memcpy_async)
- [With `cooperative_groups::memcpy_async`](#with-cooperative_groupsmemcpy_async)
- [Asynchronous Data Copies using `cuda::barrier`](#asynchronous-data-copies-using-cudabarrier)
- [Performance Guidance for `memcpy_async` (TBD)](#performance-guidance-for-memcpy_async-tbd)
  - [Alignment](#alignment)
  - [Trivially copyable](#trivially-copyable)
  - [Warp Entanglement - Commit](#warp-entanglement---commit)
  - [Warp Entanglement - Wait](#warp-entanglement---wait)
  - [Warp Entanglement - Arrive-On](#warp-entanglement---arrive-on)
  - [Keep Commit nad Arrive-On Operations Converged](#keep-commit-nad-arrive-on-operations-converged)
- [References](#references)

<br>

CUDA 11부터 `memcpy_async`라는 API를 통해 비동기 데이터 연산을 지원한다. 이 연산은 device code에서 명시적인 비동기 데이터 복사를 관리할 수 있도록 해준다. `memcpy_async`는 CUDA 커널이 연산과 데이터 이동을 오버랩할 수 있도록 해준다.

# `memcpy_async` API

`memcpy_async` APIs는 `<cuda/barrier>`, `<cuda/pipeline>`, `<cooperative_groups/memcpy_async.h>` 헤더에서 제공된다.

`cuda::memcpy_async` APIs는 `cuda::barrier`와 `cuda::pipeline`이라는 synchronization primitives와 함께 동작하는 반면, `cooperative_groups::memcpy_async`는 `cooperative_groups::wait`를 사용하여 동기화한다.

이 API들은 매우 유사한 의미를 갖는다. 다른 스레드에서 수행되는 것처럼 `src`로부터 `dst`로 객체를 복사하고, 복사가 완료되면 `cuda::pipeline`, `cuda::barrier`, 또는 `cooperative_groups::wait`를 통해 동기화될 수 있다.

> `cuda::barrier`와 `cuda::pipeline`에 대한 `cuda::memcpy_async` overload에 대한 API 문서는 [libcudacxx API](https://nvidia.github.io/libcudacxx/) 문서에서 몇 가지 예제와 함께 제공된다.

> `cooperative_groups::memcpy_async`에 대한 내용은 [Cooperative Groups](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#collectives-cg-memcpy-async) 챕터에서 자세히 다루고 있다.

> `memcpy_async` API들은 `cuda::barrier`와 `cuda::pipeline`을 사용하는데, 이들은 compute capability 7.0 이상의 GPU에서 지원된다. Compute capability 8.0 이상인 경우, global memory에서 shared memory로의 `memcpy_async` 연산은 하드웨어 가속의 이점을 누릴 수 있다.

# Copy and Compute Pattern - Staging Data Through Shared Memory

CUDA 어플리케이션은 보통 다음과 같은 _copy and compute_ 패턴을 사용한다.

1. fetches data from global memory
2. stores data to shared memory, and
3. performs computations on shared memory data, and potentially writes results back to global memory

포스팅의 나머지 부분에서 `memcpy_async` 기능을 사용 또는 사용하지 않고 이 패턴을 어떻게 표현할 수 있는지 살펴볼 예정이다.

# Without `memcpy_async`

`memcpy_async`를 사용하지 않으면, _copy and compute_ 패턴에서 복사는 `shared[local_idx] = global[global_idx]` 형태로 효현된다. Global to Shared memory 복사는 global memory로부터 register로의 read, 이어서 register로부터 shared memory의 write로 확장된다.

이 패턴이 반복적인 알고리즘에서 발생할 때, 각 스레드 블록은 `shared[local_idx] = global[global_idx]` 할당 이후에 동기화되어야 한다. 그래야 shared memory로의 모든 writes가 compute 단계가 시작되기 전에 완료된다는 것이 보장되기 때문이다. 또한, 스레드 블록은 compute 단계가 완료된 후에 다시 동기화되어야 하는데, 다음 반복 연산을 준비하기 위해 shared memory를 overwrite하는데 모든 스레드에서 연산이 끝나기 전에 overwrite가 발생하면 안되기 때문이다. 이 패턴을 코드로 표현하면 다음과 같다.

```c++
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__
void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory
}

__global__
void without_memory_async(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    size_t local_idx = block.thread_rank();

    for (size_t batch = 0; batch < batch_size; batch++) {
        // Compute the index of the current batch for this block in global memory
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        size_t global_idx = block_batch_idx + local_idx;
        shared[local_idx] = global_in[global_idx];

        block.sync(); // Wait for all copies to compute

        compute(global_out + block_batch_idx, shared); // Compute and write result to global memory

        block.sync(); // Wait for compute using shared memory to finish
    }
}
```

# With `cooperative_groups::memcpy_async`

`memcpy_async`를 사용하면, 아래와 같은 global memory to shared memory 할당은

```c++
shared[local_idx] = global_in[global_idx];
```

는 아래와 `cooperative_groups`로부터의 비동기 복사 연산으로 대체된다.

```c++
cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * block.size());
```

여기서 사용된 `cooperative_groups::memcpy_async` API는 `global_in + batch_idx`에서 시작하는 global memory를 `sizeof(int) * block.size()` 바이트 크기만큼 `shared`로 복사한다. 이 연산은 마치 다른 스레드에서 수행되는 것처럼 발생하며, 복사가 완료된 이후에 현재 스레드의 `cooperative_groups::wait` 호출로 동기화한다. 복사 연산이 완료될 때까지 global data를 수정하거나 shared data를 쓰는 행위는 race condition을 발생시킨다.

앞서 언급했듯이 compute capability 8.0 이상의 GPU에서 global to shared memory의 `memcpy_async` 전송은 하드웨어 가속의 이점을 누릴 수 있으며, 중간 레지스터를 통한 데이터 전송을 피할 수 있다.

이에 대한 코드는 다음과 같다.

```c++
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__device__
void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory
}

__global__
void with_memory_async(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        // Whole thread-group cooperatively copies whole batch to shared memory:
        cg::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size());

        cg::wait(block); // Joins all threads, waits for all copies to complete

        compute(global_out + block_batch_idx, shared); // Compute and write result to global memory

        block.sync(); // Wait for compute using shared memory to finish
    }
}
```

# Asynchronous Data Copies using `cuda::barrier`

`cuda::barrier`에 대한 `cuda::memcpy_async` overload를 사용하면 `barrier`를 이용한 비동기 데이터 전송을 동기화할 수 있다. 마찬가지로 이 구현은 마치 barrier에 바인딩된 다른 스레드에 의해 수행되는 것처럼 복사 연산을 수행한다. `barrier` 생성 시에 현재 단계의 expected count의 수를 증가시키고 복사 연산이 완료될 때 이를 감소시킨다. 따라서, 해당 `barrier`에 참여하는 모든 스레드가 도달하고 `barrier`의 현재 단계에 바인딩된 모든 `memcpy_async`가 완료된 경우에만 `barrier`의 각 단계가 진행된다.

아래 예제 코드는 block-wide `barrier`를 사용하는데, 여기서 블록의 모든 스레드가 참여한다. 이 코드는 바로 이전 예제 코드에서 `cg::wait`를 `receive_and_wait`로만 바꾸면서 이전 예제 코드와 동일한 기능을 제공한다.

```c++
#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

__device__
void compute(int* global_out, int const* shared_in) { /* ... */ }

__global__
void with_barrier(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    // Create a synchronization object
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0) {
        init(&barrier, block.size()); // Friend function initializes barrier
    }
    block.sync();

    for (size_t batch = 0; batch < batch_sz; batch++) {
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        cuda::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size(), barrier);

        barrier.arrive_and_wait(); // Waits for all copies to complete

        compute(global_out + block_batch_idx, shared);

        block.sync();
    }
}
```

> `barrier`에 대한 내용은 [Asynchronous Barrier](/cuda/doc/01_programming_guide/07-26_asynchronous_barrier.md)를 참조

# Performance Guidance for `memcpy_async` (TBD)

> 아래 내용은 아직 이해가 완전히 되지 않아서, 정리를 일단 중단한 상태. 차후 이해를 한 다음에 정리할 예정.

<details>
<summary>fold</summary>
Compute capability 8.X에서 파이프라인 메커니즘은 동일한 워프 내의 스레드 간 공유된다. 이러한 공유로 인하여 `memcpy_async` 배치가 워프 내에서 얽히게 되어 특정 상황에서는 성능에 영향을 미칠 수 있다.

이 섹션에서는 _commit_, _wait_, _arrive_ 연산에 대한 warp-entaglement 효과를 중심으로 살펴본다. 개별 동작은 [Asynchronous Data Copies using `cuda::pipeline`](/cuda/doc/01_programming_guide/07-28_asynchronous_data_copies_using_cuda_pipeline.md)을 참조.

## Alignment

Compute capability 8.0에서 [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) 명령어들은 global to shared memory로 데이터를 비동기로 복사한다. 이 명령어들에서는 한 번에 4/8/16 바이트를 복사할 수 있다. 만약  `memcpy_async`에 제공된 크기가 4/8/16의 배수이고 전달된 포인터가 4/8/16 alighment boundary로 정렬되어 있다면, `memcpy_async`는 비동기 메모리 연산만 사용하여 실행될 수 있다.

또한, `memcpy_async`를 사용할 때 최상의 성능을 달성하려면 shared memory와 global memory가 모두 128바이트로 정렬되어야 한다.

정렬에 대한 요구사항인 1 또는 2인 타입 값에 대한 포인터의 경우, 포인터가 항상 더 높은 alignment boundary에 정렬되어 있다고 증명하는 것이 불가능한 경우가 많다. `cp.async` 명령어를 사용할 수 있는지 여부를 결정하는 작업은 런타임에서 가능하지만, 이러한 런타임에서의 정렬 검사를 수행하면 코드 크기가 증가하고 런타임 오버헤드가 추가된다.

`cuda::aligned_size_t<size_t Align>(size_t size)`([link](https://nvidia.github.io/libcudacxx/extended_api/shapes/aligned_size_t.html))는 `memcpy_async`에 전달된 두 포인터가 `Align` 크기의 alignment boundary에 정렬되고 해당 크기가 `Align`의 배수라는 증거를 제공하는데 사용할 수 있다. 정렬이 올바르지 않은 경우에 대한 동작은 정의되어 있지 않다.
```c++
cuda::memcpy_async(group, dst, src, cuda::aligned_size_t<16>(N * block.size()), pipeline);
```

## Trivially copyable

만약 `memcpy_async`로 전달된 포인터 타입이 [TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable) 타입을 가리키지 않는다면 각 output element의 복사 생성자를 호출해야 하며, 이는 `memcpy_async`를 가속화하는데 사용할 수 없다.

## Warp Entanglement - Commit

`memcpy_async` 배치의 시퀀스는 워프 전체에서 공유된다. Commit 연산을 호출하는 모든 수렴 스레드에 대해 시퀀스가 한 번 증가되도록 commit 연산은 병합된다. 만약 워프가 완전히 수렴되면 시퀀스가 1씩 증가하며, 워프가 완전히 분기(diverged)되면 시퀀스는 32만큼 증가된다.

예시를 통해 살펴보자.

- _PB_ - warp-shared pipeline's actual sequence of batches: `PB = {BP0, BP1, BP2, ..., BPL}`
- _TB_ - a thread's perceived sequence of batches, as if the sequence were only incremented by this thread's invocation of the commit operation: `TB = {BT0, BT1, BT2, ..., BTL}`
- a thread's perceived sequence에서의 인덱스는 항상 actual warp-shared sequence의 인덱스와 같거나 이보다 크다. 모든 commit 연산이 수렴된 스레드로부터 실행되면 그 시퀀스는 동일하다: `BTn ≡ BPm` where `n <= m`

예를 들어, 워프가 완전히 분기하면,

- The warp-shared pipeline's actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`)
- The perceived sequence for each thread of this warp would be:
  - Thread 0: `TB = {0}` (`TL=0`)
  - Thread 1: `TB = {0}` (`TL=0`)
  - ...
  - Thread 31: `TB = {0}` (`TL=0`)

## Warp Entanglement - Wait

## Warp Entanglement - Arrive-On

## Keep Commit nad Arrive-On Operations Converged

</detailed>

# References

- [NVIDIA CUDA Documentations: Asynchronous Data Copies](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [libcudacxx API Documentation](https://nvidia.github.io/libcudacxx/)