# Table of Contents

- [Table of Contents](#table-of-contents)
- [Constant Memory](#constant-memory)
- [Implementing a 1D Stencil with Constant Memory](#implementing-a-1d-stencil-with-constant-memory)
- [Comparing with the Read-Only Cache](#comparing-with-the-read-only-cache)
- [References](#references)

# Constant Memory

Constant memory는 warp 내 스레드들이 모두 동일하게 액세스하면서 read-only인 데이터에 사용되는 특수 목적용 메모리이다. Constant memory는 **커널 내에서는 read-only**이지만, **host 측에서는 읽고 쓸 수 있다**.

Constant memory는 global memory와 같이 device DRAM에 상주하지만, 전용 on-chip 캐시가 있다. L1 캐시나 shared memory처럼, 각 SM에서 constant cache로 읽는 것이 constant memory에서 직접 읽는 것보다 latency가 훨씬 더 짧다. RTX3080 기준으로 총 64KB 크기의 constant memory가 있고, SM 당 캐시되는 크기는 8KB로 제한되어 있다.

> [NVIDIA 문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)에 compute capability (>= 5.0) 별로 스펙이 자세히 나와있다. 5.0부터 나열되어 있는데, constant memory의 크기는 모두 64KB로 동일하다.

Constant memory에 대한 최적화 기법은 다른 메모리와는 조금 다르다. Constant memory는 warp 내 모든 스레드들이 동일한 위치의 constant memory에 액세스할 때 가장 성능이 좋다. 만약 warp 내 스레드들이 다른 주소의 constant memory에 액세스한다면, 이 메모리 처리는 순차적으로 처리된다. 따라서, 스레드들이 액세스하는 다른 주소의 갯수만큼 선형적으로 비용이 증가하게 된다.

상수 변수는 아래의 qualifier와 함께 반드시 global scope에서 선언되어야 한다.
```
__constant__
```

Constant memory 변수는 프로그램이 종료할 때까지 존재하며 모든 커널에서 액세스할 수 있다. 그리고, device 측에서는 오직 읽기만 가능하므로, constant memory의 값은 반드시 host code에서 아래의 런타임 API를 통해 초기화해주어야 한다.
```c++
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind)
```

`cudaMemcpyToSymbol`은 `src`가 가리키는 데이터를 `symbol`로 지정된 constant memory location으로 복사한다. `kind`는 데이터가 전달되는 방향을 가리키며, 기본값은 `cudaMemcpyHostToDevice` 이다.

# Implementing a 1D Stencil with Constant Memory

수치해석학 분야에서 stencil computation은 주변 점으로부터 하나의 점의 값을 업데이트하는 함수에 적용할 수 있다. Stencil은 많은 알고리즘에서 편미분방정식을 푸는데도 사용된다.

1차원에서 한 점 `x` 주변의 nine-point stencil이 아래 위치의 값들에 어떤 함수를 적용한다고 가정해보자. 아래 그림은 nine-point stencil을 나타낸 것이다.
$$ \{x-4h, x-3h, x-2h, x-h, x, x+h, x+2h, x+3h, x+4h \} $$

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzJAcq%2Fbtr1T6LnOqu%2FrmMjkT56iDQZbUSW18oy81%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

이를 여러 공식 및 알고리즘에 적용할 수 있지만, 여기서 중요한 점은 9개의 점을 입력으로 취하고 하나의 출력을 생성한다는 것만 이해하면 된다. 예제에서 CUDA로 구현할 공식은 아래와 같다.

$$ f'(x) \approx a_1(f(x+h)-f(x-h)) + a_2(f(x+2h)-f(x-2h)) \\\\ + a_3(f(x+3h)-f(x-3h)) + a_4(f(x+4h)-f(x-4h)) $$

위와 같은 stencil computation에서 constant memory는 공식의 계수인 `a1, a2, a3, a4`에 적용할 수 있다. 이 계수들은 모든 스레드에서 동일하게 사용되며, 절대 변경되지 않는다. 따라서, read-only이며 broadcast access 패턴이므로 constant memory를 사용하기 적합하다. Warp 내 스레드들은 모두 같은 시점에 동일한 constant memory 주소를 참조하게 된다.

> 전체 코드는 [cmem_stencil.cu](/cuda/code/constant_memory/cmem_stencil.cu)를 참조

위 stencil computation 연산을 구현한 전체 커널 코드는 다음과 같다.
```c++
__global__
void stencil1DGPU(float* in, float* out, int const n)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < n) {
        // index to shared memory for stencil calculation
        int sidx = threadIdx.x + RADIUS;

        // read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS) {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        __syncthreads(); // sync to ensure all the data is available

        // apply the stencil
        float tmp = 0.f;

        #pragma unroll
        for (int i = 1; i <= RADIUS; i++) {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }
        // store the result
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}
```

먼저 shared memory를 선언한다. 스레드가 한 점에 대해 계산할 때, 주변 9개의 점에 대해 액세스한다. 따라서 warp 관점에서 봤을 때, 주변 9개의 점에 대해서는 read만 수행하고, 이는 warp 내 스레드들이 서로 공유할 수 있는 데이터이다. 따라서 불필요한 global memory access를 줄이기 위해 shared memory를 사용한다.
```c++
__shared__ float smem[BDIM + 2 * RADIUS];
```

`RADIUS`는 `x` 주변의 점 갯수를 나타낸다. 예제에서는 nine-point stencil이므로 이 값은 `4`이며, `x` 양 옆으로 4개의 점들이 연산에 사용된다. 그림으로 나타내면 아래와 같으며, 각 블록에서는 왼쪽/오른쪽 경계에 `RADIUS` 갯수만큼의 요소가 더 필요하므로 `2 * RADIUS`만큼 shared memory를 더 할당한다 (이를 halo라고 칭함).

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fyb2a8%2Fbtr1WzfgyGC%2FPqQ6yjfghzuGGiIcIQXJrk%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

커널 구현은 공식을 그대로 구현했으므로, 어렵지 않게 이해할 수 있을 것이다. 한 가지 유의할 점은 전체 입력의 첫 번째에서 `RADIUS`번째 요소와 마지막 `RADIUS`개의 요소를 계산할 때, access violation이 발생할 수 있다는 것이다. 이를 위해서 입력 데이터(`n`개의 요소)를 위한 메모리 공간을 할당할 때, 전체 데이터 갯수를 `n + 2 * RADIUS`로 설정하여 메모리를 할당하고 `RADIUS`번째 메모리부터 입력 데이터의 첫 번째 요소를 저장한다. 자세한 구현은 [cmem_stencil.cu](/cuda/code/constant_memory/cmem_stencil.cu)를 참조.

커널 구현에서 조금 눈 여겨 볼만한 부분은 `#pragma unroll`이라는 compiler directive를 사용했다는 점이다. 이는 loop의 반복 횟수를 컴파일 시간에 알 수 있을 때, 컴파일러가 자동으로 loop unrolling을 해준다.
```c++
#pragma unroll
for (int i = 1; i <= RADIUS; i++) {
    tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
}
```

다음으로 constant memory를 어떻게 사용하는지에 대해 살펴본다. 예제 코드에서는 계수들을 저장할 `coef` 변수를 다음과 같이 constant memory로 선언한다.
```c++
__constant__ float coef[RADIUS + 1];
```

Constant memory는 host 측에서만 `cudaMemcpyToSymbol` 런타임 API를 통해 수정이 가능하므로, 다음과 같은 방식으로 초기화해주고 있다.
```c++
void setupCoefConstant()
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float));
}
```

컴파일 후, 실행해보면 아래와 같은 출력을 얻을 수 있다.
```
$ ./cmem_stencil
> Stencil 1D at device 0: NVIDIA GeForce RTX 3080
> with array size: 16777216
stencil1DGPU <<< 524288,   32 >>> elapsed time: 0.419840 ms
```

# Comparing with the Read-Only Cache

GPU 장치의 각 SM에는 read-only cache가 있어서 global memory에 저장된 데이터를 read-only cache를 통해 액세스할 수 있다. Global memory read의 memory bandwidth와 다른 memory bandwidth를 갖는 read-only cahce이므로, bandwidth가 제한된 커널에서 성능상 이점을 얻을 수 있다.

> Read-only cache의 크기가 얼마인지를 따로 찾지는 못헀다.

Read-only cache를 통해 global memory를 액세스하려면 컴파일러에게 커널이 실행되는 동안 해당 데이터가 read-only라는 것을 알려주어야 하는데, 이는 내장 함수인 `__ldg`를 통해 가능하다. `__ldg`는 아래와 같은 방식으로 사용할 수 있으며, 강제로 read-only data cache를 통해 로드하도록 만들어준다.

```c++
__global__
void kernel(float* output, float* input) {
    ...
    output[idx] += __ldg(&input[idx]);
    ...
}
```

Read-only cache는 constant cache와 별도로 구분되는 캐시이다. constant cache를 통해 읽은 데이터는 상대적으로 그 크기가 제한되며 성능을 위해서 warp 내 스레드들이 동일한 시점에 같은 위치를 액세스해야 한다. 반면 read-only cache를 통해 읽을 수 있는 데이터는 constant cache보다 상대적으로 더 크며, non-uniform access도 가능하다.

아래 커널 코드는 위에서 구현한 `stencil1DGPU`를 수정해 read-only cache를 사용하도록 했다. 유일한 차이점은 `__ldg`를 사용하고, `d_coef`라는 global memory를 사용한다는 것이다.
```c++
__global__
void stencil1DReadOnly(float* in, float* out, float const* d_coef, int const n)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < n) {
        // index to shared memory for stencil calculation
        int sidx = threadIdx.x + RADIUS;

        // read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS) {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        __syncthreads(); // sync to ensure all the data is available

        // apply the stencil
        float tmp = 0.f;

        #pragma unroll
        for (int i = 1; i <= RADIUS; i++) {
            tmp += __ldg(&d_coef[i]) * (smem[sidx + i] - smem[sidx - i]);
        }
        // store the result
        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}
```

위 커널에서 계수는 global memory에 저장되어 있고, 이를 read-only cache를 통해 읽어야 하므로 커널을 호출하기 전에 계수 정보를 global memory로 초기화해주어야 한다.

> 전체 코드는 [cmem_readonly.cu](/cuda/code/constant_memory/cmem_readonly.cu) 참조

전체 코드를 컴파일 한 뒤, 실행시켜보면 다음과 같은 출력 결과를 얻을 수 있다.
```
$ ./cmem_readonly
> Stencil 1D at device 0: NVIDIA GeForce RTX 3080
> with array size: 16777216
stencil1DGPU      <<< 524288,   32 >>> elapsed time: 0.420864 ms
stencil1DReadOnly <<< 524288,   32 >>> elapsed time: 0.403616 ms
```

커널에서 사용되는 `coef` 배열은 broadcast access 패턴(uniform reads)으로 constant memory에 더 최적화되어 있는 것으로 생각되고, 이로 인해 read-only cache보다 constant memory가 더 좋은 성능을 보여줄 것이라고 예상했지만, 의외로 read-only cache가 조금 더 좋은 성능을 보여주고 있다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Read-Only Data Cache Load Function](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function)