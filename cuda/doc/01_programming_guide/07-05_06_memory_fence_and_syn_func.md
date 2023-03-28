# Table of Contents

- [Table of Contents](#table-of-contents)
- [Memory Fence Functions](#memory-fence-functions)
  - [Example: Single-Pass Reduction](#example-single-pass-reduction)
- [Synchronization Functions](#synchronization-functions)
- [References](#references)

<br>

# Memory Fence Functions

CUDA programming model은 device가 **weakly-ordered memory model**이라고 가정한다. 이는 CUDA thread에서 데이터를 write하는 순서는 반드시 프로그램 내에서 관측되는 순서는 아니라는 것을 의미한다. 따라서, 동기화(synchronization)없이 동일한 메모리 위치에 두 스레드가 read 또는 write하는 것은 undefined behavior이다.

아래 예제 코드에서 thread 1은 `writeXY()`를 실행하고, thread 2는 `readXY()`를 실행한다고 가정해보자.
```c++
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    int A = X;
}
```
두 스레드는 동일한 메모리 위치인 `X`와 `Y`를 동시에 read/write 한다. 모든 data-race는 undefined behavior이다. 따라서, `A`와 `B`의 결과는 알 수 없다.

Memory fence function은 메모리 액세스에 대한 순서를 순차적으로 일관되도록 할 수 있다. 이들 함수는 순서가 적용되는 범위(scope)는 다르지만 메모리 공간과는 독립적이다.

```c++
void __threadfence_block();
```

이 함수는 `cuda::atomic_thraed_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`과 동일하며, 아래의 동작을 보장한다.

- `__threadfence_block()`을 호출하기 전에 스레드에 의한 모든 메모리에 대한 모든 write는 `__threadfence_block()`을 호출한 이후에 해당 스레드의 블록 내 모든 스레드들에 의해 관측된다.
- `__threadfence_block()`을 호출하기 전에 스레드에 의한 모든 메모리에 대한 모든 read는 `__threadfence_block()`을 호출한 이후에 해당 스레드가 수행한 모든 read보다 먼저 수행된다.

```c++
void __threadfence();
```
위 함수는 `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)`와 동일하다. 그리고, `__threadfence()` 호출 이후에 발생하는 모든 메모리에 대한 모든 write는 `__threadfence()` 호출 이전에 해당 device의 모든 스레드에서 관측될 수 없다는 것을 보장한다.

```c++
void __threadfence_system();
```
위 함수는 `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)`와 동일하다. 그리고, `__threadfence()` 호출 이후에 발생하는 모든 메모리에 대한 모든 write는 `__threadfence()` 호출 이전에 해당 device/peer devices의 모든 스레드 또는 host 스레드에서 관측될 수 없다는 것을 보장한다.

> `__threadfence_system()`은 compute capability 2.x 이상의 device에서 지원된다.

이전 예제 코드에서 memory fence를 다음과 같이 추가해줄 수 있다.
```c++
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```
위 코드에서는 아래의 결과들이 관측될 수 있다.

- `A = 1` and `B = 2`
- `A = 10` and `B = 2`
- `A = 10` and `B = 20`

두 번째 write 이전에 첫 번째 write가 완료되는 것이 보장되기 때문에 이외의 결과는 불가능하다. 만약 thread 1과 thread 2가 서로 같은 블록에 속한다면, `__threadfence_block()`을 쓰는 것으로 충분하다. 만약 thread 1과 2가 서로 다른 블록에 속한다면, 동일한 device의 경우에는 `__threadfence()`를 사용하고 서로 다른 device의 CUDA threads라면 `__threadfence_system()`을 사용해야 한다.

<br>

일반적으로 스레드가 다른 스레드에서 생성된 데이터를 사용하는 경우에 memory fence를 사용한다. CUDA에서 제공하는 샘플 코드 중 single-pass reduction이 이에 해당한다.

Single-pass reduction에서는 각 블록이 먼저 배열의 subset의 partial reduce를 계산하고, 그 결과를 global memory에 저장한다. 모든 블록에서 partial reduce가 완료되면, 마지막 블록은 partial reduce 결과를 읽고 마지막 결과를 계산한다. 이때, 어떤 블록이 마지막으로 완료되었는지 확인하기 위해서 각 블록에서는 카운터를 atomic으로 증가시켜, 해당 블록에서의 계산이 끝났다는 것을 알린다. 만약 partial reduce 결과를 저장하는 코드와 카운트를 증가시키는 코드 사이에 fence가 없다면 partial reduce 결과가 저장되기 전에 카운트가 증가될 수 있다.

Memory fence functions은 오직 스레드의 메모리 연산들의 순서에만 영향을 미친다. 따라서, 스레드 간의 메모리 연산이 visible한지 여부는 보장하지 않는다. 아래 예제 코드에서는 `result` 변수에 대한 메모리 연산의 visible을 보장하기 위해 `volatile` 한정자를 선언하여 사용한다.

```c++
__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__global__ void sum(const float* array, unsigned int N,
                    volatile float* result)
{
    // Each block sums a subset of the input array.
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {

        // Thread 0 of each block stores the partial sum
        // to global memory. The compiler will use
        // a store operation that bypasses the L1 cache
        // since the "result" variable is declared as
        // volatile. This ensures that the threads of
        // the last block will read the correct partial
        // sums computed by all other blocks.
        result[blockIdx.x] = partialSum;

        // Thread 0 makes sure that the incrementation
        // of the "count" variable is only performed after
        // the partial sum has been written to global memory.
        __threadfence();

        // Thread 0 signals that it is done.
        unsigned int value = atomicInc(&count, gridDim.x);

        // Thread 0 determines if its block is the last
        // block to be done.
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    // Synchronize to make sure that each thread reads
    // the correct value of isLastBlockDone.
    __syncthreads();

    if (isLastBlockDone) {

        // The last block sums the partial sums
        // stored in result[0 .. gridDim.x-1]
        float totalSum = calculateTotalSum(result);

        if (threadIdx.x == 0) {

            // Thread 0 of last block stores the total sum
            // to global memory and resets the count
            // varialble, so that the next kernel call
            // works properly.
            result[0] = totalSum;
            count = 0;
        }
    }
}
```

## Example: Single-Pass Reduction

CUDA 샘플 코드에서 제공하는 [threadFenceReduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/threadFenceReduction)을 기반으로 unrolling 기법을 적용한 single-pass reduction을 [reduce_fence.cu](/cuda/code/reduce_integer/reduce_fence.cu)에 구현했다. 이 코드를 통해서 single-pass reduction이 어떻게 구현되는지 살펴보자.

우선 스레드 블록 하나에 대해서 reduction을 수행하는 device function이다.
```c++
__device__
void reduceBlock(volatile int* sdata, int my_sum, const unsigned int tid, cg::thread_block cta)
{
    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    // block-wide warp reduce
    int local_sum = warpReduce(my_sum);

    // save warp sum to shared memory
    if (lane_idx == 0)
        sdata[warp_idx] = local_sum;
    cg::sync(cta);

    // last warp reduce
    if (threadIdx.x < warpSize)
        local_sum = (threadIdx.x < SMEMDIM) ? sdata[lane_idx] : 0;
    
    if (warp_idx == 0)
        local_sum = warpReduce(local_sum);
    
    // write result for this block to global memory
    if (cta.thread_rank() == 0) {
        sdata[0] = local_sum;
    }
}
```
여기서는 warp shuffle instruction을 사용하여 `warpReduce` 함수 내에서 warp 내 reduce를 수행한다.

> warp shuffle reduction은 [Warp Shuffle](/cuda/study/16_warp_shuffle.md#parallel-reduction-using-the-warp-shuffle-instruction)을 참조 바람

아래 코드는 single-pass parallel reduction을 수행하는 커널 함수이다. 이전에 작성한 reduction 커널 함수와 속도를 비교하기 위해서 unrolling 기법을 적용하였다. 또한, 내부에서 마지막으로 연산을 완료한 스레드 블록을 쿼리하기 위해서 count 변수를 `__device__`로 선언하여 사용하고 있다.
```c++
__device__ unsigned int count = 0;

__global__
void reduceSinglePass(int* g_in, int* g_out, unsigned int n)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ int smem[SMEMDIM];

    // phase 1: process all inputs assigned to this block
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    if (idx >= n) return;

    // block-wide warp reduce
    int local_sum = g_in[idx];
    local_sum += g_in[idx + blockDim.x];
    local_sum += g_in[idx + blockDim.x * 2];
    local_sum += g_in[idx + blockDim.x * 3];

    reduceBlock(smem, local_sum, tid, cta);
    
    // store partial sum in a thread block
    if (tid == 0)
        g_out[blockIdx.x] = smem[0];
    
    // phase 2: last block finished will process all partial sums
    if (gridDim.x > 1) {
        __shared__ bool amLast;

        // wait until all outstanding memory instruction in this thread block are finished
        __threadfence();

        // thread 0 takes a ticket
        if (tid == 0) {
            unsigned int ticket = atomicInc(&count, gridDim.x);
            // if the ticket ID is equal to the number of blocks,
            // we are the last block
            amLast = (ticket == gridDim.x - 1);
        }

        cg::sync(cta);

        // the last block sums the results of all other blocks
        if (amLast) {
            int i = tid;
            int my_sum = 0;

            while (i < gridDim.x) {
                my_sum += g_out[i];
                i += blockDim.x;
            }

            reduceBlock(smem, my_sum, tid, cta);

            if (tid == 0) {
                g_out[0] = smem[0];
                // reset count for next run
                count = 0;
            }
        }
    }
}
```
연산 과정을 크게 2단계로 나눌 수 있는데, 먼저, 각 스레드 블록 내 reduction을 수행하게 된다. 그런 다음, 각 스레드 블록의 첫 번째 스레드에서 계산한 스레드 블록 내 부분합을 global memory에 저장한다.

다음 단계에서는 각 스레드 블록의 결과를 종합하는데, 이는 마지막으로 연산이 끝난 스레드 블록이 수행하게 된다. 어떤 스레드 블록이 마지막 블록인지 체크하기 위해서 `atomicInc`를 사용하여 `count`를 체크한다. `atomicInc`이 각 스레드 블록에서 한 번씩 호출될 때마다 `count`의 값은 1씩 증가하게 되고, 마지막 스레드 블록이 `atomicInc`를 수행하면 `count` 변수의 값은 `gridDim.x`와 같을 것이다. `atomicInc` 함수는 증가를 수행할 주소의 이전 값을 리턴하는데, 마지막 스레드 블록에서의 `atomicInc` 호출은 `gridDim.x - 1`을 리턴하게 된다. 따라서, `ticket == gridDim.x - 1`을 비교하여 마지막 스레드 블록인지 체크한다.

> `atomicInc`의 동작은 [atomicInc()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc) 참조

마지막 스레드 블록에서는 각 스레드 블록이 계산한 부분합을 모두 더한다. 그리고 `reduceBlock`을 호출하여 마지막 스레드 블록 내에서 reduction을 수행하여, 최종 결과를 계산하고 global memory array의 첫 번째 주소 위치에 저장한다.

이렇게 구현한 커널은 host 측에서 아래와 같이 호출한다. 최종 결과는 다른 reduction 커널과 달리 host 측에서 각 스레드 블록 합을 계산할 필요가 없다.
```c++
cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
reduceSinglePass<<<grid.x / 4, block>>>(d_in, d_out, num_elements);
cudaMemcpy(&gpu_sum, d_out, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduceSinglePath       : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x / 4, block.x);
```

Multi-pass로 구현된 다른 reduction 커널들과 비교하기 위해서 `nsight system`을 통해 각 커널들의 실행 시간을 측정했다. 이때, 자세한 비교를 위해 메모리 복사 시간까지 포함시켰다. 시간 측정은 `nvtx`를 통해 구간을 설정하여 측정하였다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwWjnM%2Fbtr6FhHJroY%2F0cqiDIk0QwvHKEmeHAuu30%2Fimg.png" width=800px style="display: block; margin: 0 auto"/>

대부분의 시간이 input 배열을 device로 복사하는 시간이며, 각 커널 실행 시간과 메모리 복사 시간까지 포함한 시간을 정리하면 다음과 같다.
|Kernel| Kernel Execution Time | Memory Copy Time | Total Time|
|--|--:|--:|--:|
|`reduceGmem`|0.248 ms|6.025 ms|6.273 ms|
|`reduceSmem`|0.201 ms|5.055 ms|5.251 ms|
|`reduceGmemUnroll`|0.120 ms|4.697 ms|4.817 ms|
|`reduceSmemUnroll`|0.097 ms|4.522 ms|4.619 ms|
|`reduceShfl`|0.154 ms| 4.585 ms|4.739 ms|
|`reduceShflUnroll`|0.096 ms|4.538 ms|4.634 ms|
|`reduceSinglePass`|0.115 ms|4.547 ms|4.662 ms|

사실 메모리 복사 시간이 대부분을 차지하고 있고, 그 중에서도 input 배열의 복사가 대부분을 차지한다. 마지막 reduction 결과를 host 측으로 다시 복사하는 시간은 그리 오래 걸리지 않았다 (최대 10us 소요).

여기서 눈 여겨 볼 부분은 `reduceSinglePass`의 경우에는 최종 결과까지 모두 GPU에서 연산한다는 것이다. 위 프로파일링 결과에서 각 커널 함수의 결과를 다시 host로 복사하는 비용이 그리 크지는 않았지만, reduction 결과가 이후에 GPU에서 사용된다면 다시 GPU로 복사하는 비용이 생긴다는 것이다. 반면, `reduceSinglePass`를 사용하는 경우, 최종 결과를 계산하기 위해서 host로 복사하는 연산이 필요없기 때문에 커널이 수행하는 시간이 약간 더 길어지더라도 메모리 복사 시간을 줄여서 결과적으로 더 짧은 시간 내에 연산을 끝낼 수도 있다 (`reduceSinglePass`의 경우 커널 실행 시간도 준수한 편인 것 같다).

<br>

# Synchronization Functions

```c++
void __syncthreads();
```
위 내장 함수는 한 스레드 블록 내 모든 스레드들이 이 함수 호출 지점에 도달할 때까지 대기하도록 하며, `__syncthreads()` 이전에 이 스레드들로부터 발생한 global, shared memory 액세스가 이 블록의 모든 스레드에 visible하도록 한다. 즉, `__syncthreads()`는 같은 블록 내 스레드들간의 통신을 조정하는데 사용된다 (잠재적인 data hazards를 해결할 수 있음).

조건 코드에서의 `__syncthreads()` 사용은 전체 스레드 블록 내에서 동일하게 조건이 평가되는 경우에만 허용된다. 이외의 경우에는 코드 실행이 중단되거나 의도치 않은 사이드 이펙트가 발생할 수 있다.

Compute capability 2.x 이상의 device에서는 다양한 버전의 `__synchthreads()`를 지원한다. 기본적으로 `__syncthreads()`의 동작을 수행하면서 부가적인 기능을 제공한다.

```c++
int __syncthreads_count(int predicate);
```
위 함수는 모든 스레드들이 `predicate`를 평가하고 0이 아닌 `predicate`인 스레드의 갯수를 반환한다.

```c++
int __syncthreads_and(int predicate);
```
위 함수는 모든 스레드에 대해 `predicate`가 0이 아니라고 평가되는 경우에만 0이 아닌 값을 반환한다.

```c++
int __syncthreads_or(int predicate);
```
위 함수는 모든 스레드에 대해 `predicate`가 하나라도 0이 아니라고 판단되는 경우, 0이 아닌 값을 반환한다.

```c++
int __syncwarp(unsigned mask=0xffffffff);
```
위 함수는 `mask`에 지정된 모든 warp lanes이 이 함수를 실행할 때까지 대기하도록 한다. 각 스레드마다 자신의 bit set이 있어야 하며, 모든 스레드들은 동일한 mask로 이를 실행해야 한다.

<br>

# References

- [NVIDIA CUDA Documentations: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- [NVIDIA CUDA Documentations: Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
- CUDA Sample Code: [threadFenceReduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/threadFenceReduction)