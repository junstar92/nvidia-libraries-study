# Table of Contents

- [Table of Contents](#table-of-contents)
- [The Warp Shuffle Instruction](#the-warp-shuffle-instruction)
- [Variants of the Warp Shuffle Instruction](#variants-of-the-warp-shuffle-instruction)
  - [\_\_shfl\_sync()](#__shfl_sync)
  - [\_\_shfl\_up\_sync()](#__shfl_up_sync)
  - [\_\_shfl\_down\_sync()](#__shfl_down_sync)
  - [\_\_shfl\_xor\_sync()](#__shfl_xor_sync)
- [Sharing Data within a Warp](#sharing-data-within-a-warp)
  - [Broadcast of a Value across a Warp](#broadcast-of-a-value-across-a-warp)
  - [Shift Up within a Warp](#shift-up-within-a-warp)
  - [Shift Down within a Warp](#shift-down-within-a-warp)
  - [Shift within a Warp with Warp Around](#shift-within-a-warp-with-warp-around)
  - [Butterfly Exchange across the Warp](#butterfly-exchange-across-the-warp)
  - [Exchange Values of an Array across a Warp](#exchange-values-of-an-array-across-a-warp)
  - [Exchange Values Using Array Indices across a Warp](#exchange-values-using-array-indices-across-a-warp)
- [Parallel Reduction Using the Warp Shuffle Instruction](#parallel-reduction-using-the-warp-shuffle-instruction)
- [References](#references)

<br>

# The Warp Shuffle Instruction

Compute Capability 3.0 이상의 GPU에서는 **shuffle** instructions을 사용할 수 있다. Shared memory를 사용하면 스레드 블록 내의 스레드 간의 통신을 low-latency로 수행할 수 있다. Shuffle instruction은 동일한 warp 내의 두 스레드에서 한 스레드가 다른 스레드의 레지스터(register)를 직접 읽을 수 있는 메커니즘이다.

Shuffle instruction을 사용하면 스레드가 shared memory 또는 global memory를 거치지 않고 서로 직접 데이터를 교환할 수 있으며, shared memory보다 latency가 짧으며 추가적인 메모리를 소비하지도 않는다. 따라서, shuffle instruction은 warp 내 스레드 간에 데이터를 빠르게 교환할 수 있는 방법을 제공한다.

Shuffle instruction은 warp 내 스레드 간에 수행된다. 이때, 이 스레드들을 구분하기 위해 **lane** 이라는 개념을 도입한다. Lane은 단순히 warp 내 단일 스레드를 나타내며, warp의 크기가 32이므로 각 lane은 0~31 인덱스로 식별된다. 당연히 스레드 블록 내 warp는 여러 개가 있을 수 있기 때문에, warp 내 각 스레드는 고유한 lane 인덱스를 갖지만, 스레드 블록 내의 여러 스레드들이 동일한 lane 인덱스를 가질 수 있다. 이는 그리드 내에서 스레드들이 서로 같은 `threadIdx.x` 값을 가질 수 있다는 것과 동일하다. 다만, lane 인덱스를 위한 내장 변수는 없으므로, 직접 계산해주어야 한다. 1D thread block에서 주어진 스레드에 대한 lane 인덱스 및 warp 인덱스는 아래와 같이 계산할 수 있다 (warp 인덱스는 스레드 블록 내 warp들을 식별하기 위한 인덱스이다).
```c++
lane_idx = threadIdx.x % 32;
warp_idx = threadIdx.x / 32;
```

예를 들어, 한 스레드 블록 내의 thread 1과 thread 33은 동일한 lane 인덱스 1이지만, 서로 다른 warp 인덱스(각각 0과 1)를 갖는다. 2D 스레드 블록이라면, 2차원 스레드 좌표를 1차원 스레드 인덱스로 변환한 뒤, 위의 공식을 이용하여 lane과 warp의 인덱스를 계산할 수 있다.

<br>

# Variants of the Warp Shuffle Instruction

> CUDA 9.0부터 모든 GPU에서 `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor`은 deprecated 이다. Compute capability 7.x 이상에서는 위 함수들은 더 이상 사용할 수 없으며, 위 함수에 `_sync`가 붙은 버전을 사용해야 한다 (ex, `__shfl_sync`).

다양한 데이터 타입에 대해서 4개의 warp shuffle instruction을 제공한다.
```c++
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```

`T`로 가능한 타입은 다음과 같다.
- `int`, `unsigned int`
- `long`, `unsigned long`
- `long long`, `unsigned long long`
- `float`
- `double`
- `__half`, `__half2` (included in `cuda_fp16.h`)
- `__nv_bfloat16`, `__nv_bfloat162` (included in `cuda_bf16.h`) 

## __shfl_sync()

먼저 `__shfl_sync`를 살펴보자.
```c++
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
```
이 함수는 같은 warp 내에서 `srcLane`로 식별되는 스레드가 전달한 `var`값을 반환한다. 이를 사용하면 동일한 warp 내 각 스레드들이 특정 스레드의 값을 가져올 수 있다. 이 연산은 warp 내 모든 active thread에 대해 동시에 발생하며, 4바이트 또는 8바이트의 데이터를 이동시킨다.

옵셔널로 `width` 변수에는 2~32사이의 2의 거듭제곱 수를 설정할 수 있다. 기본값은 `warpSize = 32`이며, warp 내 모든 스레드에서 shuffle instruction이 수행된다. 이때, `srcLane`은 값을 가져올 source thread의 lane ID이다. 만약 `width`의 값이 32가 아니라면 각 스레드의 lane ID와 연산이 수행되는 shuffle ID가 반드시 같지는 않다. 만약 `width`가 32가 아닌 다른 값이라면 1D thread block에서 각 스레드의 shuffle ID는 아래와 같이 계산할 수 있다.
```c++
shuffle_id = threadIdx.x % width;
```

마지막으로 함수의 첫 번째 파라미터로 전달하는 `mask`가 있다. 아직 정확히 이해하지는 못했는데, 문서 내에서는 이에 대해 아래와 같이 언급하고 있다.
```
The new *_sync shfl intrinsics take in a mask indicating the threads participating in the call. A bit, representing the thread’s lane id, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. Each calling thread must have its own bit set in the mask and all non-exited threads named in mask must execute the same intrinsic with the same mask, or the result is undefined.
```
일단, shuffle instruction을 호출하는데 참여하는 스레드들을 나타내는 것으로 보인다. 각 비트가 스레드의 lane ID를 나타낸다고 한다. 비트값을 0으로 설정한다고 하더라도 shuffle instruction 수행을 막는 등의 강제성은 없는 것 같다. 아래 예제 코드에서는 `mask`값을 모두 `0xffffffff`로 설정했다.

`__shfl_sync`를 아래와 같이 호출해보자.
```c++
int value = __shfl_sync(0xffffffff, x, 3, 16);
```
위와 같이 호출하면 0~15까지의 스레드들은 스레드 3으로부터 x의 값을 받고, 16~31까지의 스레드들은 스레드 19로부터 x의 값을 받게된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcSFiwU%2Fbtr3SwaGOje%2FY4KVCJYTxarByPg2EUxllK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

## __shfl_up_sync()

Shuffle operation의 또 다른 버전은 상대적으로 낮은 ID의 스레드로부터 데이터를 복사하는 연산이다.
```c++
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
```
이 함수는 source lane index를 이 함수를 호출한 스레드의 lane 인덱스에서 `delta`를 뺀 값으로 설정한다. 따라서, 이 연산을 호출하면 `var`값이 `delta`만큼 오른쪽으로 shift하게 된다. `__shfl_up_sync`로 커버되지 않는 스레드가 존재하게 되는데, 이러한 스레드의 값을 변경되지 않는다 (파라미터로 전달한 값을 그냥 그대로 반환한 셈이다). 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdtyqlP%2Fbtr3ID9OYWJ%2F4qETljkyknyPzvRckyTMkk%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

## __shfl_down_sync()

`__shfl_down_sync`는 상대적으로 높은 ID의 스레드로부터 데이터를 복사하는 연산이다.
```c++
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
```

이 함수에서 source lane index는 이 함수를 호출한 스레드의 lane 인덱스에서 `delta`를 더해서 계산한다. 즉, `__shfl_up_sync`와 반대로 `var`값이 `delta`만큼 왼쪽으로 shift하게 된다. 마찬가지로 커버되지 않는 스레드들의 값은 변경되지 않는다. 그림으로 표현하면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fr4Img%2Fbtr3TGDQDaJ%2FiIibVB92h2kgqLjDrswKOk%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

## __shfl_xor_sync()

Shuffle instruction의 마지막 버전은 caller의 lane ID와 bitwise XOR로 계산한 lane으로부터 데이터를 전달한다.
```c++
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```
이 함수는 source lane index를 caller의 lane ID와 `laneMask`를 bitwise XOR 연산을 수행하여 계산한다. 이를 사용하면 아래 그림과 같이 butterfly addressing pattern을 쉽게 수행할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBBTUw%2Fbtr3U4qYj5i%2FF8Wq4bFhDdpKH4eoLEYzgk%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

<br>

# Sharing Data within a Warp

이번에는 예제를 통해 warp shuffle instruction을 사용하면 어떤 연산들을 수행할 수 있는지 살펴보자. 예제 코드에서는 모든 커널이 16개의 스레드로 구성된 하나의 1D thread block으로 호출된다. 그리고 16개의 요소로 구성된 배열에 대해 warp shuffle instruction을 수행하며, 각 요소의 값은 해당 요소의 인덱스 값으로 초기화되며, 각 warp shuffle instruction을 수행하면 해당 값들이 어떻게 변경되는지 확인한다.

> 전체 예제 코드는 [simple_shfl.cu](/cuda/code/warp_shuffle/simple_shfl.cu)를 참조

## Broadcast of a Value across a Warp

아래 커널은 warp-level broadcast를 구현한 것이다.
```c++
__global__
void test_shfl_broadcast(int* d_out, int* d_in, int const src_lane)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff, value, src_lane, BDIMX);
    d_out[threadIdx.x] = value;
}
```

예제 코드에서는 위 커널을 아래의 코드로 호출한다.
```c++
test_shfl_broadcast<<<1, block>>>(d_out, d_in, 2);
```

이 코드를 실행한 결과는 다음과 같다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle broadcast       :  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
```

2번 lane ID의 값으로 broadcast되어 모든 값들이 2가 된 것을 확인할 수 있다.

## Shift Up within a Warp

아래 커널 구현은 shuffle shift-up operation을 구현한 것이다. 즉, 배열의 데이터를 오른쪽으로 shift한다.
```c++
__global__
void test_shfl_up(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_up_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}
```

위 커널의 `delta`를 2로 설정하여 실행한 결과는 다음과 같다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle up              :  0  1  0  1  2  3  4  5  6  7  8  9 10 11 12 13
```

각 스레드의 값들이 오른쪽으로 2칸씩 이동한 것을 볼 수 있다. 가장 왼쪽의 2개의 스레드는 자신보다 상대적으로 왼쪽에 있는 스레드가 없기 때문에 값이 변하지 않는다.

> 완전한 rotation 연산에 대해서는 [Shift within a Warp with Warp Around](#shift-within-a-warp-with-warp-around)를 참조

## Shift Down within a Warp

아래 커널은 반대로 데이터를 왼쪽으로 shift하는 shuffle shift-down operation 커널이다.
```c++
__global__
void test_shfl_down(int* d_out, int* d_in, unsigned int const delta)
{
    int value = d_in[threadIdx.x];
    value = __shfl_down_sync(0xffffffff, value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}
```

마찬가지로 `delta`값을 2로 지정하여 실행하면 아래의 결과를 얻을 수 있다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle down            :  2  3  4  5  6  7  8  9 10 11 12 13 14 15 14 15
```
가장 오른쪽 두 개의 스레드에서는 자신보다 상대적으로 높은 land ID의 스레드가 없기 때문에 값이 변경되지 않는다.

## Shift within a Warp with Warp Around

`test_shfl_up` 커널과 `test_shfl_down` 커널은 데이터를 shift 하지만, 전체 배열이 회전하지는 않는다. 따라서 변경되지 않는 부분이 있다. 아래 커널 구현은 이를 보완한 shift warp-around operation을 구현한 것이다.

```c++
__global__
void test_shfl_wrap(int* d_out, int* d_in, int const offset)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff, value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
}
```

이전 구현들에서 `delta`는 positive만 가능했지만, 위 커널의 `offset`은 negative도 가능하다. 위 커널 내에서 shuffle instruction의 source lane ID는 자신의 lane ID에 `offset`을 더해 계산된다.

`offset`의 값을 2로 지정하여 실행하면 아래의 결과를 얻을 수 있고 (왼쪽으로 rotation),
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle wrap left       :  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1 
```
`offset`의 값을 -2로 지정하면 아래의 결과를 얻을 수 있다 (오른쪽으로 rotation).
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle wrap right      : 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13
```

## Butterfly Exchange across the Warp

다음 커널은 두 스레드 간의 butterfly addressing pattern을 구현한다. 값이 교환되는 스레드는 이를 호출하는 스레드의 lane ID와 `mask` 간 bitwise XOR로 계산된다.
```c++
__global__
void test_shfl_xor(int* d_out, int* d_in, int const mask)
{
    int value = d_in[threadIdx.x];
    value = __shfl_xor_sync(0xffffffff, value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}
```

`mask`의 값을 1로 지정하면, 아래의 결과를 얻을 수 있다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle xor 1           :  1  0  3  2  5  4  7  6  9  8 11 10 13 12 15 14
```

간단히 설명하기 위해서, lane ID가 4와 5인 스레드를 예시로 살펴보자.

Lane ID 4(`0b0100`)와 `mask`값인 1(`0b0001`)에 대해 bitwise XOR 연산을 수행하면, 5(`0b0101`)가 된다. 반대로 Lane ID 5(`0b0101`)와 `mask` 1(`0b0001`)에 대해 bitwise XOR 연산을 수행하면, 4(`0b0100`)가 된다. 따라서, lane ID가 4와 5인 스레드의 값이 서로 교환되는 것이다. 나머지 스레드에 대해서 계산해보면 왜 위와 같이 인접한 두 스레드가 서로 값을 교환하는지 알 수 있다.


## Exchange Values of an Array across a Warp

연속된 몇몇 스레드의 값을 동일한 갯수의 다른 몇몇 스레드의 값과 교환할 수도 있다. 아래 커널은 이러한 동작을 구현한다.
```c++
#define SEGM 4

__global__
void test_shfl_xor_array(int* d_out, int* d_in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    value[0] = __shfl_xor_sync(0xffffffff, value[0], mask, BDIMX);
    value[1] = __shfl_xor_sync(0xffffffff, value[1], mask, BDIMX);
    value[2] = __shfl_xor_sync(0xffffffff, value[2], mask, BDIMX);
    value[3] = __shfl_xor_sync(0xffffffff, value[3], mask, BDIMX);

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}
```

위 커널 구현은 `SEGM` 값이 4로 지정되어 있는데, 연속된 4개의 스레드의 값들을 인접하면서 연속된 4개의 스레드의 값들과 교환한다. 즉, 이를 실행하면, 다음의 결과를 얻을 수 있다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle xor array 1     :  4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11 
```

16개의 스레드에서의 값들을 4개씩 묶어서 교환한다. 이때, 하나의 스레드가 연속된 4개의 값들을 커버하게 되며, 총 16개의 값을 4개씩 교환하기 때문에 스레드 블록은 16개가 아닌 4개의 스레드로만 실행한다. 따라서, 위 커널은 아래와 같이 execution configuration을 지정하여 호출해야 한다.
```c++
test_shfl_xor_array<<<1, block.x / SEGM>>>(d_out, d_in, 1);
```

위의 실행 결과에서 `mask`의 값은 1로 지정했다. 따라서, lane ID가 0인 스레드는 lane ID가 1인 스레드와 4개의 값들을 교환(`0,1,2,3<->4,5,6,7`)하고, lane ID가 2인 스레드는 lane ID가 3인 스레드와 4개의 값들을 교환(`8,9,10,11<->12,13,14,15`)하게 된다.

## Exchange Values Using Array Indices across a Warp

지금까지 살펴본 shuffle operation 커널 함수들은 `offset`을 통해 배열에서 일정한 위치에 있는 값들을 가져오게 된다. 만약 `offset`이 아닌 서로 다른 위치의 값들을 가지고 오려면 shuffle instruction을 기반으로 하는 swap 함수가 필요하다. 즉, 처음 4개의 데이터에서 첫 번째 데이터와 그 다음 4개의 데이터에서 마지막 데이터만 서로 교환하고 싶은 경우가 이에 해당한다.

이를 위한 커널 구현은 다음과 같다.
```c++
__inline__ __device__
void swap(int* value, int lane_idx, int mask, int first_idx, int second_idx)
{
    bool pred = ((lane_idx / mask + 1) % 2 == 1);

    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }

    value[second_idx] = __shfl_xor_sync(0xffffffff, value[second_idx], mask, BDIMX);

    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }
}

__global__
void test_shfl_swap(int* d_out, int* d_in, int const mask, int first_idx, int second_idx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        value[i] = d_in[idx + i];

    swap(value, threadIdx.x, mask, first_idx, second_idx);

    #pragma unroll
    for (int i = 0; i < SEGM; i++)
        d_out[idx + i] = value[i];
}
```

전체적인 흐름은 `test_shfl_xor_array` 커널과 비슷하지만, `test_shfl_swap` 커널은 `first_idx`와 `second_idx`를 추가로 받는다. 여기서 `first_idx`는 처음 4개의 데이터에서 바꾸고자하는 데이터의 인덱스이며, `second_idx`는 처음 4개의 데이터에 대응하는 다른 4개의 데이터에서 바꾸고자하는 데이터의 인덱스이다.

만약, 아래와 같이 인자를 설정하여 커널을 호출하게 되면,
```c++
test_shfl_swap<<<1, block.x / SEGM>>>(d_out, d_in, 1, 0, 3);
```
다음과 같이 결과를 얻게 된다.
```
initial data            :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shuffle xor swap        :  7  1  2  3  4  5  6  0 15  9 10 11 12 13 14  8
```
0과 7, 8과 15가 서로 교환된 것을 볼 수 있다. 즉, [0, 1, 2, 3]에서 0번째 인덱스인 0과 [4, 5, 6, 7]에서 3번째 인덱스인 7이 서로 교환되었다. 어떻게 이렇게 동작하는지는 `swap` device 함수를 이해하면 알 수 있다.

`swap` 함수에 대해 간단히 설명하면 다음과 같다. 먼저, 처음 4개의 데이터에서 `first_idx`와 `second_idx`에 위치하는 값을 서로 바꾼다. 그러면 처음 4개의 데이터에서 바꿀 데이터와 이에 대응하는 다음 4개의 데이터에서 바꿀 데이터의 위치가 `second_idx`로 동일하게 된다. 결과적으로 동일한 offset에 대해 shuffle instruction을 수행하게 되는 것이다. 이렇게 값을 교환한 뒤에는 다시 처음 4개의 데이터에 대해 `first_idx` 위치의 값과 `second_idx` 위치의 값을 바꿔주면, 결과적으로 [0, 1, 2, 3 / 4, 5, 6, 7]에서 0과 7의 위치만 바뀌게 된다.

<br>

# Parallel Reduction Using the Warp Shuffle Instruction

[Avoiding Branch Divergence](/cuda/study/07_avoiding_branch_divergence.md), [Unrolling Loops](/cuda/study/08_unrolling_loops.md), 그리고 [Reducing Global Memory Access](/cuda/study/12-2_reducing_global_memory_access.md)에서 sum reduction을 구현하는 여러 가지 방법들에 대해 살펴봤었다. 여기서 shared memory롤 최적화한 방법이 가장 빠르다는 것을 확인했다.

Reduction 연산의 경우에는 warp shuffle instruction으로도 구현할 수 있는데, 기본 아이디어는 매우 심플하며 여기에는 3단계의 reduction이 포함된다.

- Warp-level reduction
- Block-level reduction
- Grid-level reduction

하나의 스레드 블록은 여러 개의 warp들로 구성된다. Warp-level reduction에서 각 warp는 자신의 데이터에 대해 reduction을 수행한다. 이때, shared memory를 사용하지 않고, 각 스레드는 먼저 레지스터를 사용하여 하나의 데이터를 저장한다.
```c++
int local_sum = g_in[idx];
```

그리고 warp-level reduction을 수행한다. 우선 warp 내 스레드 간 reduction은 inline 함수로 다음과 같이 구현할 수 있다.
```c++
__inline__ __device__
int warpReduce(int local_sum)
{
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 16, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 8, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 4, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2, warpSize);
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1, warpSize);

    return local_sum;
}
```
위 함수를 통해 계산된 각 warp의 `local_sum`을 shared memory에 저장한다. 이 과정은 아래와 같이 구현할 수 있다.
```c++
// calculate lane index and warp index
int lane_idx = threadIdx.x % warpSize;
int warp_idx = threadIdx.x / warpSize;

// block-wide warp reduce
int local_sum = warpReduce(g_in[idx]);

// save warp sum to shared memory
if (lane_idx == 0)
    smem[warp_idx] = local_sum;
```

다음으로 block-level reduction을 수행한다. Warp-level reduction 단계의 마지막에서 각 warp의 `local_sum`을 shared memory에 저장한다. Shared memory에 저장된 `local_sum`을 warp ID가 0인 warp의 레지스터로 다시 복사하고, 해당 warp에서만 block-level reduction을 수행한다. 따라서, 스레드 블록 내 동기화가 먼저 필요하다. 이 과정을 구현하면 다음과 같다.
```c++
__syncthreads();

// last warp reduce
if (threadIdx.x < warpSize)
    local_sum = (threadIdx.x < SMEMDIM) ? smem[lane_idx] : 0;

if (warp_idx == 0)
    local_sum = warpReduce(local_sum);

// write result for this block to global memory
if (tid == 0)
    g_out[blockIdx.x] = local_sum;
```
Block-level reduction 결과는 마지막에 global memory로 저장된다. Grid-level reduction은 host 측에서 for문을 통해 수행되어 최종 결과를 계산한다.

전체 커널 구현은 다음과 같다.
```c++
__global__
void reduceShfl(int* g_in, int* g_out, unsigned int const n)
{
    // shared memory for each warp sum in a thread block
    __shared__ int smem[SMEMDIM];

    // boundary check
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    // block-wide warp reduce
    int local_sum = warpReduce(g_in[idx]);

    // save warp sum to shared memory
    if (lane_idx == 0)
        smem[warp_idx] = local_sum;
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize)
        local_sum = (threadIdx.x < SMEMDIM) ? smem[lane_idx] : 0;
    
    if (warp_idx == 0)
        local_sum = warpReduce(local_sum);
    
    // write result for this block to global memory
    if (tid == 0)
        g_out[blockIdx.x] = local_sum;
}
```

> 전체 코드는 [reduce_shfl.cu](/cuda/code/warp_shuffle/reduce_shfl.cu)에서 확인할 수 있음

Global memory, shared memory, unrolling 기법 등으로 구현한 커널들과 함께 실행한 결과는 다음과 같다.
```
$ ./reduce_shfl
> Starting reduction at device 0: NVIDIA GeForce RTX 3080
> Array size: 16777216
cpu reduce             : 5.1149 ms     cpu sum: 2139353471
reduceGmem             : 0.2556 ms     gpu sum: 2139353471 <<<grid 65536 block 256>>>
reduceSmem             : 0.1950 ms     gpu sum: 2139353471 <<<grid 65536 block 256>>>
reduceGmemUnroll       : 0.1264 ms     gpu sum: 2139353471 <<<grid 16384 block 256>>>
reduceSmemUnroll       : 0.1036 ms     gpu sum: 2139353471 <<<grid 16384 block 256>>>
reduceShfl             : 0.1505 ms     gpu sum: 2139353471 <<<grid 65536 block 256>>>
reduceShflUnroll       : 0.1022 ms     gpu sum: 2139353471 <<<grid 16384 block 256>>>
```

Unrolling 기법은 적용하지 않고 shared memory를 사용한 커널(`reduceSmem`)보다 warp shuffle instruction을 사용한 커널(`reduceShfl`)의 실행 속도가 더 빠르다는 것을 확인할 수 있다.

Warp shuffle instruction에 unrolling 기법을 적용(`reduceShflUnroll1`)하면, 가장 빠른 속도를 얻을 수 있었다. 다만, shared memory를 사용한 커널(`reduceSmemUnroll`)와 눈에 띄는 차이는 없었다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [NVIDIA Forums: What does mask mean in warp shuffle functions](https://forums.developer.nvidia.com/t/what-does-mask-mean-in-warp-shuffle-functions-shfl-sync/67697)
- [Stackoverflow: Insight into the first argument mask in __shfl_sync()](https://stackoverflow.com/questions/58833808/insight-into-the-first-argument-mask-in-shfl-sync)