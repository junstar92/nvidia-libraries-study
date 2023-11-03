# Table of Contents

- [Table of Contents](#table-of-contents)
- [Single-Stage Asynchronous Data Copies using `cuda::pipeline`](#single-stage-asynchronous-data-copies-using-cudapipeline)
- [Multi-Stage Asynchronous Data Copies using `cuda::pipeline`](#multi-stage-asynchronous-data-copies-using-cudapipeline)
- [Pipeline Interface](#pipeline-interface)
- [Pipeline Primitives Interface](#pipeline-primitives-interface)
- [References](#references)

<br>

CUDA에서는 데이터의 이동과 연산을 비동기로 관리하고 오버랩하기 위한 `cuda::pipeline` 동기화 객체를 제공한다.

`cuda::pipeline`에 대한 API 문서는 [libcudacxx API](https://nvidia.github.io/libcudacxx/)에서 확인할 수 있다. Pipeline 객체는 head와 tail을 갖는 double-ended N stage queue (deq 구조)이며, FIFO 순서로 작업을 처리하는데 사용된다. Pipeline 객체는 아래의 멤버 함수를 가지고 파이프라인의 스테이지를 관리한다.

| Pipeline Class Memeber Function | Description |
|--|--|
| `producer_acquire`| 파이프라인 내부 큐에서 이용 가능한 스테이지를 얻는다 |
| `producer commit` | 현재 얻은 파이프라인 스테이지에서 `producer_acquire` 호출 이후 발행된 비동기 연산을 commit 한다 |
| `consumer_wait` | 파이프라인의 가장 오래된 스테이지에서의 모든 비동기 명령이 완료될 때까지 기다린다 |
| `consumer_release` | 재사용을 위해 파이프라인의 가장 오래된 스테이지를 파이프라인에 릴리즈한다. 그러면 릴리즈된 스테이지는 producer에 의해 acquire할 수 있다 |

# Single-Stage Asynchronous Data Copies using `cuda::pipeline`

[Asynchronous Barrier](/cuda/doc/01_programming_guide/07-26_asynchronous_barrier.md)와 [Asynchronous](/cuda/doc/01_programming_guide/07-27_asynchronous_data_copies.md)에서 `cooperative_groups`와 `cuda::barrier`를 사용하여 비동기 데이터 전송을 하는 방법에 대해서 살펴볼 수 있었다. 이번에는 하나의 스테이지에서 `cuda::pipeline` API를 사용하여 비동기 복사를 스케쥴링하는 방법에 대해서 살펴본다. 이후에는 여러 스테이지를 사용하여 연산과 복사를 오버랩하는 방법을 살펴본다.

예제 코드는 다음과 같다.
```c++
#include <cooperative_groups/memory_async.h>
#include <cuda/pipeline>

namespace cg = cooerative_groups;

__device__ void compute(int* global_out, int const* shared_in);
__global__
void with_single_stage(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 1; // pipeline with one stage
    // One batch must fit in shared memory
    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    // allocate shared storage for a two-stage cuda::pipeline
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements
    // Compute offset of the batch `batch` of this thread block in global memory
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t batch = 0; batch < batch_sz; batch++) {
        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared, global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations committed to the
        // previous 'compute' stage to complete
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }
}
```

# Multi-Stage Asynchronous Data Copies using `cuda::pipeline`

이전 예제에서 커널의 스레드는 shared memory로의 데이터 전송이 완료될 때까지 대기한다. Shared memory로 전달하기 전에 global memory를 register로 데이터를 전송하는 것은 피하지만 `memcpy_async` 연산의 latency를 연산과 오버랩하여 hiding하지는 않는다.

이번 예제에서는 CUDA `pipeline`의 특징을 제대로 활용한다. Pipeline은 `memcpy_async` batches의 시퀀스를 관리하는 메커니즘을 제공하고 CUDA 커널이 메모리 전송과 연산을 오버랩할 수 있도록 해준다. 아래 예제 코드는 2-stage pipeline을 구현하여 데이터 전송과 연산을 오버랩한다. 코드에는 다음의 내용들이 구현된다:

- Initialize the pipeline shared state
- Kickstarts the pipeline by scheduling a `memcpy_async` for the first batch
- Loops over all the batches: it schedules `memcpy_async` for the next batch, blocks all threads on the completion of the `memcpy_async` for the previous patch, and overlaps the computation on the previous batch with the asynchronous copy of the memory for the next batch
- Finally, it drains the pipeline by performing the computation on the last batch

```c++
#include <cooperative_groups/memory_async.h>
#include <cuda/pipeline>

namespace cg = cooerative_groups;

__device__ void compute(int* global_out, int const* shared_in);
__global__
void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // pipeline with two stages
    // Two batches must fit in shared memory
    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each batch

    // Allocate shared storage for a two-stage cuda::pipeline
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();

    // Pipelined copy/compute
    for (size_t batch = 1; batch < batch_sz; batch++) {
        // Stage indices for the compute and copy stages
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;

        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations commited to the
        // previous 'compute' stage to complete
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the 'copy' stage
        compute(global_out + global_idx, shared + shared_offset[compute_staage_idx]);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }

    // Compute the data fetch by the last iteration
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz - 1), shared + shared_offset[(batch_sz - 1) % 2]);
    pipeline.consumer_release();
}
```

> 코드에서 루프 내 `compute` 부분이 잘못된 것으로 추정된다. Loop 진입 전의 비동기 복사는 [0, block.size()] 위치의 global memory에 대한 값이 복사되었으므로, loop의 첫 번째 반복에서 `compute` 스테이지는 `global_out`의 [0, block.size()] 위치에 대한 연산이 이루어져야 하는데, 코드에서는 다음 단계의 위치에서 `global_out`을 가리키고 있다. `global_out + global_idx`는 `global_out + block_batch(batch - 1)`이 되어야 할 것 같다.

`pipeline` 객체는 _head_ 와 _tail_ 을 갖는 덱(dequeue)이며 FIFO 순으로 작업을 처리하는데 사용된다고 언급했었다. 생산자(producer) 스레드는 pipeline의 head로 작업을 commit하는 반면, 소비자(consumer) 스레드는 pipeline의 tail로부터 작업을 pull한다. 위 예제 코드에서 모든 스레드는 생산자 이자 소비자이다. 스레드는 이전 batch의 `memcpy_async` 연산이 완료되기를 기다리는 동안, 먼저 다음 batch를 위한 `memcpy_async` 연산을 commit한다.

- 파이프라인 스테이지에 작업을 커밋하는 방법은 다음과 같다.
  - `pipeline.producer_acquire()`를 사용하여 생상자 스레드 집합으로부터 파이프라인 _head_ 를 얻는다 (acquire).
  - 파이프라인 head에 `memcpy_async` 연산을 submit한다.
  - `pipeline.producer_commit()`을 사용하여 파이프라인 head에 commit(advance)한다.
- 이전에 커밋된 스테이지를 사용하는 방법은 다음과 같다.
  - `pipeline.consumer_wait()`를 사용하여 _tail_ (oldest) 스테이지를 기다린다 (스테이지가 완료될 때까지 기다린다).
  - `pipeline.consumer_release()`를 사용하여 스테이지를 _release_ 한다.

`cuda::pipeline_shared_stage<scope, count>`는 `count` 갯수 만큼 동시 스테이지를 파이프라인이 처리할 수 있도록 유한한 리소스를 캡슐화한다. 만약 모든 리소스가 사용되고 있다면, `pipeline.producer_acquire()`는 다음 파이프라인의 리소스가 소비자 스레드에 의해서 release될 때까지 생산자 스레드를 블럭시킨다.

<br>

아래 예제 코드는 방금 예제 코드에서 루프의 프롤로그(prolog) 및 에필로그(epilog)와 루프 자체를 병합하여 보다 간결하게 작성한 것이다.
```c++
template<size_t stages_count = 2> // Pipeline with stages_count stages
__global__
void with_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_sz * grid.size()); // assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count]; // Offsets to each batch
    for (int s = 0; s < stages_count; s++) shared_offset[s] = s * block.size();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    // compute batch: next batch to process
    // fetch_batch: next batch to fetch from global memory
    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_size; compute_batch++) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); fetch_batch++) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            size_t block_batch_idx = block_batch(batch_idx);
            cuda::memcpy_async(block, shared + shared_offset[shared_idx], global_in + block_batch_idx, sizeof(int) * block.size(), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```

<br>

위 예제 코드에서 사용된 `pipeline<thread_scope_block>` 프리미티브는 매우 유연하다. 위 예제 코드에서는 사용되지 않았지만 아래의 두 가지 기능을 지원한다.

- any arbitary subset of threads in the block can participate in the `pipeline`.
- from the threads that participate, any subsets can be producers, consumers, or both.

아래 예제 코드는 짝수 랭크의 스레드는 생산자로, 홀수 랭크의 스레드는 소비자로 사용하는 방법을 보여준다.

```c++
__device__ void compute(int* global_out, int shared_in);

template<size_t stages_count = 2>
__global__
void with_specialized_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // In this example, threads with "even" thread rank are producers, while threads with "odd" thread rank are consumers
    const cuda::pipeline_role thread_role
        = block.thread_rank() % 2 == 0 ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    
    // Each thread block only has half of its threads as producer
    auto producer_threads = block.size() / 2;

    // Map adjacent even and odd threads to the same id;
    const int thread_idx = block.thread_rank() / 2;

    auto elements_per_batch = size / batch_sz;
    auto elements_per_batch_per_block = elements_per_batch / grid.group_dim().x;

    extern __shared__ int shared[]; // stages_count * elements_per_batch_per_block * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; s++) shared_offset[s] = s * elements_per_batch_per_block;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

    // Each thread block processes `batch_sz` batches.
    // Compute offset of the batch `batch` of this thread block in global memory.
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; compute_batch++) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); fetch_batch++) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            if (thread_role == cuda::pipeline_role::producer) {
                // Only the producer threads schedule asynchronous memcpy
                pipeline.producer_acquire();
                size_t shared_idx = fetch_batch % stages_count;
                size_t batch_idx = fetch_batch;
                size_t global_batch_idx = block_batch(batch_idx) + thread_idx;
                size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
                cuda::memcpy_async(shared + shared_batch_idx, global_in + global_batch_idx, sizeof(int), pipeline);
            }
        }
        if (thread_role == cuda::pipeline_role::consumer) {
            // Only the consumer threads compute
            pipeline.consumer_wait(0);
            size_t shared_idx = compute_batch % stages_count;
            size_t global_batch_idx = block_batch(compute_batch) + thread_idx;
            size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
            compute(global_out + global_batch_idx, *(shared + shared_batch_idx));
            pipeline.consumer_release();
        }
    }
}
```

예를 들어, 모든 스레드가 생산자와 소비자인 경우, `pipeline`이 수행하는 몇 가지 최적화가 있지만 일반적으로 이러한 모든 기능을 지원하는 비용을 완전히 없앨 수는 없다. 예를 들어, `pipeline`은 동기화를 위해 shared memory에 일련의 barrier를 저장하고 사용한다. 이는 블록에 모든 스레드가 파이프라인에 참여하는 경우 필요하지 않다.

파이프라인에서 블록 내 모든 스레드가 참여하는 특별한 케이스에서는 `__syncthreads()`와 함께 `pipeline<thread_scope_thread>`를 사용하면 `pipeline<thread_scope_block>`보다 더 좋을 수 있다. 예제 코드는 다음과 같다.
```c++
template<size_t stages_count>
__global__
void with_staging_scope_thread(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto thread = cg::this_thread();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    // No pipeline::shared_state needed
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; compute_batch++) {
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); fetch_batch++) {
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            // Each thread fetches its own data:
            size_t thread_batch_idx = block_batch(batch_idx) + threadIdx.x;
            // The copy is performed by a single `thread` and the size of the batch is now that of a single element:
            cuda::memcpy_async(thread, shared + shared_offset[shared_idx] + threadIdx.x, global_in + thread_batch_idx, sizeof(int), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync(); // __syncthreads(): All memcpy_async of all threads in the block for this stage have completed here
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```

만약 `compute` 연산이 동일한 워프에 있는 다른 스레드가 write한 shared memory만을 읽는다면 `__syncwarp()`만으로도 충분하다.

# Pipeline Interface

`cuda::memcpy_async`에 대한 API 문서는 [libcudacxx API Documentation](https://nvidia.github.io/libcudacxx/)에서 확인할 수 있다.

`pipeline` 인터페이스는 아래의 조건을 만족해야 한다.

- at least CUDA 11.0
- at least SIO C++ 2011 compatibility, e.g., to be compiled with `-std=c++11`
- `#include <cuda/pipeline>`

만약 ISO C++ 2011 호환성없이 컴파일하는 경우, C-like interface가 있으며 [Pipeline Primitives Interface](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#pipeline-primitives-interface)에서 확인할 수 있다.

# Pipeline Primitives Interface

Pipeline 프리미티브는 `memcpy_async` 기능에 대한 C-like 인터페이스이다. 이 인터페이스는 `<cuda_pipeline.h>` 헤더를 통해 사용할 수 있다. ISO C++ 2011 호환성없이 컴파일할 때는 `<cuda_pipeline_primitives.h>`를 include하면 된다.

> 파이프라인 프리미티브에 대한 내용은 [Pipeline Primitives Interface](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#pipeline-primitives-interface)를 참조

# References

- [NVIDIA CUDA Documentations: Asynchronous Data Copie using `cuda::pipeline`](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-pipeline)
- [libcudacxx API Documentation](https://nvidia.github.io/libcudacxx/)