# Table of Contents

- [Table of Contents](#table-of-contents)
- [Simple Synchronization Pattern](#simple-synchronization-pattern)
- [Temporal Splitting and Five Stages of Synchronization](#temporal-splitting-and-five-stages-of-synchronization)
- [Bootstrap Initialization, Expected Arrival Count, and Participation](#bootstrap-initialization-expected-arrival-count-and-participation)
- [A Barrier's Phase: Arrival, Countdown, Completion, and Reset](#a-barriers-phase-arrival-countdown-completion-and-reset)
- [Spartial Partioning (also known as Warp Specialization)](#spartial-partioning-also-known-as-warp-specialization)
- [Early Exit (Dropping out of Participation)](#early-exit-dropping-out-of-participation)
- [Completion function](#completion-function)
- [Memory Barrier Primitives Interface](#memory-barrier-primitives-interface)
- [References](#references)

<br>

NVIDIA C++ 표준 라이브러리에서는 `std::barrier`의 GPU 구현이 있다. `std::barrier` 구현과 함께 라이브러리에서는 사용자가 barrier objects의 scope를 지정할 수 있는 extension을 제공한다. Barrier API scopes는 [Thread Scopes](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes) 문서에서 정의하고 있다. Compute capability 8.0 이상의 GPU에서는 barrier operations에 대한 하드웨어 가속과 `memcpy_async` 기능과의 통합을 제공한다. Compute capability 7.0 이상, 8.0 미만인 디바이스에서는 하드웨어 가속없이 barrier를 사용할 수 있다.

# Simple Synchronization Pattern

arrive/wait barrier를 사용하지 않고도 `__syncthreads()` 또는 [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)에서 제공하는 `group.sync()`를 통해 동기화를 달성할 수 있다 (to synchronize all threads in a block).

```c++
#include <cooperative_group.h>

using cg = cooperative_groups;

__global__
void simple_sync(int iteration_count)
{
    auto block = cg::this_thread_block();

    for (int i = 0; i < iteration_count; i++) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

스레드들은 동기화 지점(`block.sync()`)에서 모든 스레드들이 이 지점에 도달할 때까지 블럭된다. 또한, 동기화 지점 이전에 발생한 메모리 업데이트는 동기화 지점 이후의 모든 스레드에서 visible 하다는 것이 보장된다.

> `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)`과 동작이 동일하다.

이 패턴은 3 단계로 구성된다.

1. Code **before** sync performs memory updates that will be read **after** the sync
2. Synchronization point
3. Code **after** sync point with visibility of memory updates that happened **before** sync point

# Temporal Splitting and Five Stages of Synchronization

`std::barrier`를 사용한 temporally-split synchronization 패턴은 다음과 같다.

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__
void compute(float* data, int curr_iteration);

__global__
void split_arrive_wait(int iteration_count, float* data)
{
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; curr_iter++) {
        /* code before arrive */
        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
        compute(data, curr_iter);
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive() */
        /* code after wait */
    }
}
```

위 패턴에서 동기화 지점(`block.sync()`)는 arrive point(`bar.arrive()`)와 wait point(`bar.wait(std::move(token))`)으로 분할된다. 한 스레드는 `bar.arrive()`에 대한 첫 번째 호출로 `cuda::barrier`에 참여하기 시작한다. 스레드가 `bar.wait(std::move(token))`을 호출할 때, 참여한 스레드가 `init()`에 전달된 arrival count 인자에 지정된 예상 횟수만큼 `bar.arrive()`를 완료할 때까지 블럭된다. 참여한 스레드들의 `bar.arrive()` 호출 전에 발생하는 메모리 업데이트는 `bar.wait(std::move(token))` 호출 이후에 참여한 스레드들에 visible이 보장된다. 주목해야할 점은 `bar.arrive()` 호출은 스레드를 블럭하지 않으며, 참여하는 다른 스레드의 `bar.arrive()` 호출 이전에 발생하는 메모리 업데이트에 의존하지 않는 작업을 진행할 수 있다.

이러한 _arrive and then wait_ 패턴은 5단계로 구성되며 이는 반복될 수 있다.

1. Code **before** arrive performs memory updates that  will be read **after** the wait
2. Arrive point with implicit memory fence (i.e., equivalent to `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)`)
3. Code **between** arrive the wait
4. Wait point
5. Code **after** the wait, with visibility of updates that were performed **before** the arrive

# Bootstrap Initialization, Expected Arrival Count, and Participation

모든 스레드는 `cuda::barrier`에 참여하기 전에 초기화가 발생해야 한다.

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__
void init_barrier() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Single thread initializes the total expected arrival count
    }
    block.sync();
}
```

모든 스레드들이 `cuda::barrier`에 참여하기 전에 barrier는 `init()`과 예상되는 arrival count와 함께 반드시 초기화되어야 한다. 위 예제 코드에서는 expected arrival count는 `block.size()`로 주어졌다. 초기화는 반드시 스레드가 `bar.arrive()`를 호출하기 전에 발생해야 한다. 이는 스레드들이 `cuda::barrier`에 참여하기 전에 동기화되어야 하지만, 동기화를 위해서는 `cuda::barrier`를 만들어야 하기 때문에 bootstrapping 문제를 야기한다. 이 예제에서 참여하는 스레드들은 cooperative group의 일부이고 `block.sync()`를 사용하여 초기화를 부트스트랩한다. 이 예제에서는 하나의 스레드 블록 전체가 초기화에 참여하기 때문에 `__syncthreads()` 또한 사용될 수 있다.

`init()`의 두 번째 파라미터는 **expected arrival count**, 즉, 참여하는 스레드가 `bar.wait(std::move(token))`을 호출하여 블럭이 해제되기 전에 호출되는 `bar.arrive()`의 횟수이다. 위 예제 코드에서 `cuda::barrier`는 스레드 블록 내 스레드의 갯수(`cg::this_thread_block().size()`)로 초기화된다. 그리고 이 스레드 블록 내 모든 스레드들은 이 barrier에 참여한다.

`cuda::barrier`는 스레드가 참여하는 방식(split arrive/wait)과 어떤 스레드가 참여할 지 지정할 수 있어서 유연하다. Cooperative groups의 `this_thread_block.sync()` 또는 `__syncthreads()`는 한 스레드 블록 전체에 적용되고 `__syncwarp(mask)`는 워프(warp) 내 부분집합에 지정된다는 것과는 대조적이다. 다만, 스레드 블록 또는 전체 워프를 동기화하는 것이 목적이라면 `__syncthreads()` 및 `__syncwarp(mask)`를 사용하는 것이 성능상으로 더 좋다.

# A Barrier's Phase: Arrival, Countdown, Completion, and Reset

`cuda::barrier`는 참여하는 스레드가 `bar.arrive()`를 호출함에 따라 expected arrival count에서 0으로 카운트다운한다. 카운트다운이 0에 도달하면 현재 단계의 `cuda::barrier`가 완료된다. `bar.arrive()`의 마지막 호출로부터 카운트다운이 0에 도달하면 카운트다운은 자동적으로 리셋된다(atomically). 리셋이 되면 카운트다운은 expected arrival count으로 할당되고, `cuda::barrier`는 다음 단계로 이동한다.

`token = bar.arrive()`로부터 반환되는 `cuda::barrier::arrival_token` 클래스의 객체인 `token`은 barrier의 현재 단계와 연관되어 있다. `bar.wait(std::move(token))` 호출은 이를 호출한 스레드를 `cuda::barrier`가 현재 단계에 있는 동안 블럭한다 (즉, token과 연관된 단계가 `cuda::barrier`의 단계와 일치하는 동안). 만약 `bar.wait(std::move(token))`이 호출되기 전에 단계가 진행되면 그 스레드는 블럭되지 않는다. 또한, `bar.wait(std::move(token))`에서 스레드가 블럭된 상태에 있는 동안 단계가 진행되면, 그 스레드는 언블럭된다.

> **It is essential to knwo when a reset could or could not occur, especially in non-trivial arrive/wait synchronization patterns.**

- 한 스레드에서 `token = bar.arrive()`와 `bar.wait(std::move(token))` 호출은 `cuda::barrier`의 현재 단계 동안 `token = bar.arrive()`가 호출되도록 순차적이어야 한다. `bar.wait(std::move(token))` 호출은 동일한 단계 또는 다음 단계 동안 발생한다.
- 한 스레드에서 `bar.arrive()` 호출은 반드시 barrier의 카운트가 0이 아닐 때 일어나야 한다. Barrier 초기화 이후에 만약 스레드에서의 `bar.arrive()` 호출로 인해 카운트다운이 0이 된다면, `bar.wait(std::move(token))`를 먼저 호출해야 `bar.arrive()`에 대한 후속 호출에서 barrier를 재사용할 수 있다.
- `bar.wait()`는 반드시 현재 단계 또는 바로 이전 단계의 `token` 객체를 사용하여 호출해야 한다. 다른 값의 `token` 객체에 대한 동작은 정의되지 않는다.

# Spartial Partioning (also known as Warp Specialization)

스레드 블록은 공간적으로 분할될 수 있고, 따라서 워프를 독립적인 계산을 수행하도록 특화시킬 수 있다 (warp specialization). 공간 분할(spatial partitioning)은 생산자(producer) 또는 소비자(consumer) 패턴에서 사용될 수 있으며, 여기서 스레드의 한 하위 집합이 다른 스레드의 하위 집합에서 동시에 소비되는 데이터를 생산한다.

Producer/consumer spatial partitioning 패턴은 생산자(producer)와 소비자(consumer) 간의 데이터 버퍼를 관리하기 위해 두 개의 one sided synchronization이 필요하다.

| Producer | Consumer |
|---|---|
| wait for buffer to ready to be filled | signal buffer is ready to be filled |
| produce data and fill the buffer | |
| signal buffer is filled | wait for buffer to be filled |
| | consume data in filled buffer |

생산자 스레드는 소비자 스레드에서 버퍼가 채워질 준비가 되었다는 시그널을 기다린다. 그러나 소비자 스레드는 이 시그널을 기다리지 않는다. 소비자 스레드에서는 생산자 스레드로부터 버퍼가 채워졌다는 시그널을 기다리지만, 생산자 스레드는 마찬가지로 이 시그널을 기다리지 않는다. 전체 생산자/소비자 동시성에서 이 패턴에는 (적어도) 두 개의 `cuda::barrier`가 필요한 double buffering이 있다.

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__
void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N / buffer_len); i++) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2) */
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__
void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); i++) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

// N is the total number of float elements in arrays in and out
__global__
void producer_consumer_pattern(int N, int buffer_len, float* in, float* out)
{
    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // buffer_0 = buffer and buffer_1 = buffer + buffer_len
    __shared__ extern float buffer[];

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];

    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4) {
        init(bar + block.thread_rank(), block.size());
    }
    block.sync();

    if (block.thread_rank() < warpSize) {
        producer(bar, bar+2, buffer, in, N, buffer_len);
    }
    else {
        consumer(bar, bar+2, buffer, out, N, buffer_len);
    }
}
```

위 예제 코드에서 첫 번째 워프는 생산자로 특화되고 나머지 워프들은 소비자로 특화된다. 모든 생산자 및 소비자 스레드는 4개의 `cuda::barrier` 각각에 참여(`bar.arrive()` 또는 `bar.arrive_and_wait()` 호출)하므로 expected arrival count는 `block.size()`와 동일하다.

생산자 스레드는 소비자 스레드로부터 shared memory buffer가 채워질 수 있다는 시그널을 기다린다. 이때, 생산자 스레드는 token을 얻기 위해 먼저 `ready[i%2].arrive()`에 도달한 다음 token을 사용하는 `ready[i%2].wait(token)`에 도달한다. 이 연산은 `ready[i%2].arrive_and_wait()`로 결합하여 사용할 수 있다. 즉, `bar.arrive_and_wait()`는 `bar.wait(bar.arrive())`와 동일하다.

생산자 스레드들은 ready buffer를 계산하고 채운다. 그런 다음 filled barrier(`filled[i%2].arrive()`)에 도달하여 버퍼가 채워졌다는 시그널을 보낸다. 이 시점에서 생산자 스레드는 대기하지 않는다. 대신 다음 반복의 버퍼(double buffering)가 채워질 준비가 될 때까지 기다린다.

소비자 스레드는 두 버퍼가 채워질 준비가 되었다는 시그널을 보내는 것으로 시작된다. 소비자 스레드는 이 시점에서 대기하지 않고, 대신 반복에서의 버퍼가 채워질 때까지 기다린다 (`filled[i%2].arrive_and_wait()`). 소비자 스레드가 버퍼를 소비한 후, 해당 버퍼가 다시 채워질 준비가 되었다고 시그널을 보낸다 (`ready[i%2].arrive()`). 그런 다음 다음 반복의 buffer가 채워질 때까지 기다린다.

# Early Exit (Dropping out of Participation)

동기화 시퀀스에 참여하고 있는 스레드가 해당 시퀀스에서 일찍 종료되어야 하는 경우, 그 스레드는 종료하기 전에 명시적으로 참여를 중단(drop out)해야 한다. 참여하는 나머지 스레드는 후속 `cuda::barrier`에 arrive 및 wait 작업을 정상적으로 진행할 수 있다.

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ bool condition_check();

__global__
void early_exit_kernel(int N)
{
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0)
        init(&bar, block.size());
    block.sync();

    for (int i = 0; i < N; i++) {
        if (condition_check()) {
            bar.arrive_and_drop();
            return;
        }
        /* other threads can proceed normally */
        barrier::arrival_token token = bar.arrive();
        /* code between arrive and wait */
        bar.wait(std::move(token));
        /* code after wait */
    }
}
```

이러한 작업은 현재 단계에서 도달해야 하는 참여 중인 스레드들이 정상적으로 작업을 진행할 수 있도록, 더 이상 도달하지 않는 스레드 갯수만큼 expected arrival count를 감소시킨다.

# Completion function

`cuda::barrier<Scope, CompletionFunction>`에서 `CompletionFunction`은 마지막 스레드가 도착한 다음, 스레드가 `wait`로부터 언블럭될 때 phase 당 한 번 실행된다. 해당 단계에서 `barrier`에 도달한 스레드가 수행한 메모리 연산들은 `CompletionFunction`을 실행하는 스레드에서 visible이며, `CompletionFunction` 내에서 수행된 모든 메모리 작업들은 `barrier`에서 대기 중인 모든 스레드가 `wait`로부터 언블럭 된 이후에 모든 스레드에서 visible이다.

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <functional>
namespace cg = cooperative_groups;

__device__ int divergent_compute(int*, int);
__device__ int independent_computation(int*, int);

__global__
void psum(int* data, int n, int* acc)
{
    auto block = cg::this_thread_block();

    constexpr int BlockSize = 128;
    __shared__ int smem[BlockSize];
    assert(BlockSize == block.size());
    assert(n % 128 == 0);

    auto completion_fn = [&] {
        int sum = 0;
        for (int i = 0; i < 128; i++) sum += smem[i];
        *acc += sum;
    }

    // Barrier storage
    // Note: the barrier is not default-construtible because
    //       completion_fn is not default-constructible due to the capture.
    using completion_fn_t = decltype(completion_fn);
    using barrier_t = cuda::barrier<cuda::thread_scope_block, completion_fn_t>;
    __shared__ std::aligned_storage<sizeof(barrier_t), alignof(barrier_t)> bar_storage;

    // Initialize barrier
    barrier_t* bar = (barrier_t*)&bar_storage;
    if (block.thread_rank() == 0) {
        assert(*acc == 0);
        assert(blockDim.x == blockDim.y == blockDim.y == 1);
        new (bar) barrier_t{block.size(), completion_fn};
        // equivalent to: init(bar, block.size(), completion_fn);
    }
    block.sync();

    // Main loop
    for (int i = 0; i < n; i += block.size()) {
        smem[block.thread_rank()] = data[i] * *acc;
        auto t = bar->arrive();
        // We can do independent computation here
        bar->wait(std::move(t));
        // shared-memory is safe to re-use in the next iteration
        // since all threads are done with it, including the one
        // that did the reduction
    }
}
```

# Memory Barrier Primitives Interface

Memory barrier primitives는 `cuda::barrier` 기능에 대한 C-like interface이다. 이들은 `<cuda_awbarrier_primitives.h>` 헤더에 정의되어 있다. 이에 대한 내용은 [문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-barrier-primitives-interface)를 참조바라며, 여기서 따로 다루지는 않는다.

# References

- [NVIDIA CUDA Documentations: Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier)