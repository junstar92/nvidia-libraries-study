# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [Shared Memory](#shared-memory)
- [Shared Memory Allocation](#shared-memory-allocation)
- [Shared Memory Banks and Access Mode](#shared-memory-banks-and-access-mode)
  - [Memory Banks](#memory-banks)
  - [Bank Conflict](#bank-conflict)
  - [Access Mode](#access-mode)
  - [Memory Padding](#memory-padding)
  - [Access Mode Configuration](#access-mode-configuration)
- [Configuring the Amount Of Shared Memory](#configuring-the-amount-of-shared-memory)
- [Synchronization](#synchronization)
  - [Weakly-Ordered Memory Model](#weakly-ordered-memory-model)
  - [Explicit Barrier](#explicit-barrier)
  - [Memory Fence](#memory-fence)
  - [Volatile Qualifier](#volatile-qualifier)
- [References](#references)

<br>

# Intro

GPU에는 다음의 두 가지 타입의 메모리가 있다.

- On-board memory
- On-chip memory

Global memory는 큰 on-board memory이며, 비교적 높은 latency를 갖는다. 반면 shared memory는 더 작지만, low-latency on-chip memory이다. 따라서, global memory보다 더 높은 bandwidth를 갖는다. Shared memory는 program-managed cache라고 생각하면 되고, 일반적으로 다음의 상황에서 유용하다.

- An intra-block thread communication channel
- A program-managed cache for global memory data
- Scratch pad memory for transforming data to improve global memory access patterns

이번 포스팅에서는 shared memory에 대해 자세히 살펴보고, 다른 포스팅들을 통해 reduction과 matrix transpose 예제에 어떻게 shared memory를 적용할 수 있는지, 그리고 성능에 어떻게 영향을 미치는지 알아본다.

# Shared Memory

Shared memory(SMEM)은 GPU에서 중요한 컴포넌트 중 하나이다. 물리적으로 각 SM은 현재 실행 중인 스레드 블록의 모든 스레드들이 공유하는 small low-latency memory pool을 포함한다. 같은 스레드 블록 내의 스레드들은 shared memory를 사용하여 서로 협력하거나 on-chip data를 재사용하고, 커널에 필요한 global memory bandwidth를 크게 줄일 수 있다. Shared memory에 내용물은 어플리케이션에서 명시적으로 관리될 수 있기 때문에 shared memory를 program-managed cache라고도 부른다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQPwq8%2Fbtr0RQcNNYB%2F7xRyIpa3Jmz19zTuc0aFx0%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

위 그림은 CUDA의 메모리 계층을 보여준다. 그림에서 볼 수 있듯이 global memory에 대한 load/store request는 L2 cache를 통과한다. Shared memory와 L1 cache는 L2 cache보다 물리적으로 SM에 더 가깝다. 결과적으로, shared memory의 latency는 global memory보다 매우 낮으며 bandwidth는 훨씬 더 크다.

각 스레드 블록이 실행될 때, 정해진 양의 shared memory가 각 스레드 블록에 할당된다. 이 shared memory의 address space는 스레드 블록의 모든 스레드에게 공유된다. 따라서, shared memory는 자신이 할당된 스레드 블록의 lifetime과 같으며, warp에 의한 shared memory에 대한 액세스 요청은 하나의 transaction으로 처리된다.

가장 최악의 경우는 warp 내에서 shared memory에 대한 액세스 요청이 순차적으로 32개의 transaction들로 수행되는 것이다. Shared memory를 액세스하는 패턴에 대해서는 [Shared Memory Banks and Access Mode](#shared-memory-banks-and-access-mode)에서 자세히 다룬다.

Shared memory는 하나의 SM에서 상주하는 모든 스레드 블록 간에 분할되는 리소스이다. 따라서, 병렬로 처리할 수 있는 정도를 제한하는 매우 중요한 리소스이다. **커널이 shared memory를 더 많이 사용할수록, 동시에 활성화될 수 있는 스레드 블록의 수는 줄어든다.**

# Shared Memory Allocation

Shared memory 변수를 할당하거나 선언하는 여러 가지 방법이 있다. 정적으로 선언하거나 동적으로 선언할 수도 있고, 커널 내에서 local로 선언하거나 CUDA source code file 내에서 global로 선언할 수 있다. CUDA는 1D, 2D, 3D shared memory 배열 선언을 지원한다.

기본적으로 shared memory 변수는 아래의 qualifier를 지정하여 선언된다.
```c++
__shared__
```

아래의 배열 선언은 2D float 배열을 shared memory로 정적으로 선언한다.
```c++
__shared__ float tile[size_y][size_x];
```
만약 커널 함수 내에서 선언한다면, 그 변수의 scope는 해당 커널에 대해 local이다. 만약 파일 내에서 커널 외부에 선언된다면, 이 변수의 scope는 모든 커널에 대해 global이다.

만약 컴파일 시간에 shared memory의 크기를 알 수 없다면, `extern` 키워드를 사용하여 크기가 지정되지 않은 배열을 선언할 수 있다. 예를 들어, 아래 코드는 1D un-sized `int` shared memory 배열을 선언한다. 이 또한 커널 내부나 외부에서 선언될 수 있다.
```c++
extern __shared__ int tile[];
```

컴파일 시간에 크기를 알 수 없기 때문에 커널을 실행할 때 그 크기를 지정하여 동적으로 shared memory를 할당해주어야 한다. 동적으로 할당할 shared memory의 크기는 `execution configuration(<<<...>>>)`의 3번째 인자로 전달하며, 단위는 `bytes`이다.
```c++
kernel<<<grid, block, num_elements * sizeof(int)>>>(...)
```

> 동적으로 shared memory를 할당할 때는 오직 1D array 선언만 가능하다.

# Shared Memory Banks and Access Mode

메모리 성능을 최적화할 때 측정해야할 두 가지 핵심 속성은 latency와 bandwidth이다. [Memory Access Pattern](/cuda-study/11_memory_access_patterns.md)에서는 여러 global memory access pattern들이 어떻게 latency와 bandwidth에 영향을 미치는지 살펴봤다. Shared memory는 global memory latency와 bandwidth 성능의 영향을 숨기기 위해 사용할 수 있다. 먼저 shared memory를 최대한 활용하기 위해서 어떻게 shared memory가 정렬되는지 이해하는 것이 중요하다.

## Memory Banks

높은 memory bandwidth를 달성하기 위해서, shared memory는 **banks**라고 불리는 32개의 동일한 크기의 memory module로 나뉘며, 이들은 동시에 액세스할 수 있다. 32개의 bank가 있는 이유는 한 warp 내에 32개의 스레드가 있기 때문이다.

Shared memory는 1D address space 이다. GPU의 compute capability에 따라 shared memory의 주소는 서로 다른 패턴으로 서로 다른 bank에 매핑된다. 이에 대한 내용은 아래에서 조금 더 다룬다. 한 warp에 의해서 실행되는 shared memory load/store operation이 한 bank당 둘 이상의 메모리 위치에 액세스하지 않는 경우, memory operation은 하나의 memory transaction으로 처리될 수 있다. 그렇지 않다면, 여러 memory transaction으로 처리되기 때문에 memory bandwidth utilization은 감소된다.

> Shared memory의 access pattern은 아래에서 자세히 다룬다.

## Bank Conflict

한 Warp 내에서 shared memory에 요청된 주소들이 동일한 memory bank에 속하면, **bank conflict**가 발생한다. Bank conflict가 발생하면 하드웨어는 conflict가 발생하지 않는 필요한 만큼의 transaction으로 분할하여 둘 이상의 별도의 conflict-free transaction으로 memory request를 처리하게 된다. 따라서, 분할된 memory transaction의 수 만큼 effective bandwidth가 감소된다.

> 조금 헷갈릴 수 있는데, 하나의 bank에는 하나의 메모리 주소만 있는 것이 아니며 여러 주소가 들어있다. 아래에서 shared memory access pattern에 대해서 살펴볼텐데, 개별 스레드들이 **동일한 shared memory bank에 있는 주소를 요청하더라도 요청한 주소가 동일한지에 따라 conflict 발생 여부가 달라진다**.

Warp에 의해 발생하는 shared memory request에서는 일반적으로 아래의 3가지 상황이 발생한다.

- **Parallel access** : multiple addresses accessed across multiple banks
- **Serial access** : multiple addresses accessed within the same bank
- **Broadcast access** : a single address read in a single bank

**Parallel access**가 가장 일반적인 패턴이다. Warp 내에서 액세스되는 주소들이 여러 bank에 속한 경우이다. 전부는 아니지만 액세스되는 주소들이 하나의 memory transaction으로 처리될 수 있음을 의미한다. 가장 최적인 경우는 모든 주소들이 서로 다른 bank에 속하는 것이며, 32개의 스레드에 의한 32개의 memory request가 하나의 memory transaction으로 처리된다.

**Serial access**는 가장 최악의 패턴이다. 요청된 여러 주소들이 같은 bank에 속하는 경우이며, memory request는 순차적으로 수행된다. 만약 warp 내 32개의 스레드들이 동일한 bank에 있는 서로 다른 메모리 주소에 액세스한다면, 32개의 memory transaction이 요청되어 최적의 상황보다 32배 증가하게 된다.

**Broadcase access**는 warp 내 모든 스레드가 동일한 bank의 동일한 주소를 읽는 경우에 해당한다. 하나의 memory transaction만 수행되고, 액세스된 word는 다른 스레들로 브로드캐스팅된다. 한 번의 memory transaction만 필요하지만, 일부 바이트만 읽으므로 bandwidth utilization은 떨어지게 된다.

아래 그림은 최적의 parallel access 패턴을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcfzdJN%2Fbtr1nEuj0Yj%2F8DMwKC38okczeelkHbcjc1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

각 스레드는 하나의 32-bit word에 액세스한다. 각 스레드는 서로 다른 bank의 주소에 액세스하기 때문에 bank conflict는 발생하지 않는다.

아래 그림은 irregular, random access 패턴을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdGaMbX%2Fbtr0NI0tXmJ%2FRIXMcV6yypuGlNxFkmdcc0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

이 경우도 각 스레드가 서로 다른 bank의 주소에 액세스하므로 bank conflict는 발생하지 않는다.

아래 그림은 또 다른 irregular access 패턴을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmC5GZ%2Fbtr0VKJ9cAv%2F9CFU6y9VLo9QpAKQxrKD31%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

이 패턴에서는 몇몇 스레드들이 같은 bank에 액세스하고 있다. 이 경우에서는 어떤 주소를 요청하느냐에 따라 아래의 두 가지 상황이 가능하다.

- Conflict-free broadcast access: 스레드들이 동일한 bank의 동일한 주소에 액세스할 때
- Bank conflict access: 스레드들이 동일한 bank의 서로 다른 주소에 액세스할 때

아래 그림들은 공식 CUDA 문서 내에서 언급하고 있는 shared memory access 패턴이다. 아래 그림에서는 같은 bank에 서로 다른 주소를 서로 다른 빨간 사각형으로 표시하고 있기 때문에 조금 더 이해하기 수월하다.

바로 아래 그림은 strided shared memory access 패턴을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbuO8nl%2Fbtr1blbBWnO%2FciI3heyt8If5A7HkTo3S31%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

- Left: Linear addressing with a stride of one 32-bit word (no bank conflict)
- Middle: Linear addressing with a stride of two 32-bit words (two-way bank conflict)
- Right: Linear addressing with a stride of three 32-bit words (no bank conflict)

아래 그림은 irregular shared memory access 패턴을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJ7uHL%2Fbtr0VKwDWFV%2FNIW0UHFKPksk6h7hziwPJk%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

- Left: Conflict-free access via random permutation
- Middle: Conflict-free access since threads 3, 4, 6, 7, and 9 access the same word(address) within bank 5
- Right: Conflict-free broadcast access (threads access the same word(address) within a bank)

## Access Mode

Shared memory bank의 size(width)는 shared memory bank에 어떤 shared memory addresses가 있는지를 정의한다. Memory bank의 size는 compute capability에 따라 다른데, 현재 두 가지의 bank size가 가능하다.

- 4 bytes (32-bits)
- 8 bytes (64-bits)

> CUDA 문서에서 이에 대한 내용은 찾을 수가 없었고, 오래된 NVIDIA Developer Blog에서 bank size를 설정하는 방법에 대해 간단히 언급을 하고 있긴 하다 ([link](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)). Blog 내용에 따르면 8 bytes 크기로 bank size를 설정하여 double precision data에 액세스할 때 발생하는 bank conflict를 해결할 수 있다고 한다.

우선 shared memory의 access mode를 한 마디로 설명하면, 메모리 주소가 bank에 매핑되는 방법이라고 할 수 있다. 현재 RTX3080(compute capability 8.6)을 사용하여 테스트를 하고 있는데, 최근 GPU에서의 default bank size는 4 bytes로 확인이 된다. 이는 아래의 간단한 코드를 실행시켜 확인해볼 수 있다.

```c++
#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("> Device %d: %s\n", dev, prop.name);

    cudaSharedMemConfig smem_config;
    cudaDeviceGetSharedMemConfig(&smem_config);
    printf("> Default Shared Memory Size: %s\n",
        smem_config == cudaSharedMemConfig::cudaSharedMemBankSizeFourByte ? "32-bits" : "64-bits");

    return 0;
}
```

RTX3080에서 위 코드는 다음과 같이 출력된다.
```
> Device 0: NVIDIA GeForce RTX 3080
> Default Shared Memory Size: 32-bits
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFHC85%2Fbtr1nCXPz64%2Fbq3ZhkHk4oBLuqRYSmNuT1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

위 그림과 같이 32-bit mode에서는 바이트 주소를 4로 나누어 4-byte word index로 변환할 수 있다. 그리고 word index는 32개의 bank에 아래와 같이 bank index로 매핑된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcHoCL3%2Fbtr1dceF9TZ%2FGcH4UYg22KGtxYtJKWijrK%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

위에서 언급했듯이 동일한 warp의 두 스레드가 동일한 주소에 액세스할 때는 bank conflict가 발생하지 않는다. 이러한 메모리 액세스가 read인 경우에는 word가 이를 요청한 스레드로 브로드캐스팅된다. 하지만 write인 경우에는 이 메모리를 요청한 스레드들 중 하나에서만 word에 write를 수행한다. 이때, 어떤 스레드가 write를 수행하는지는 정의되지 않는다.

64-bit mode에서는 연속적인 64-bit words가 연속된 bank에 매핑된다. 각 뱅크는 64bits/clock의 bandwidth를 갖게 된다.

Warp내 두 스레드가 동일한 64-bit word 내 sub-word에 액세스하는 경우, 이를 처리하기 위해서는 하나의 64-bit read만 필요하므로 bank conflict는 발생하지 않는다. 결과적으로 동일한 액세스 패턴에 대해서 64-bit(8-byte) mode는 항상 32-bit(4-byte) mode와 같거나 더 적은 bank conflict를 발생시킨다.

32-bit mode에서는 연속된 32-bit words가 연속된 bank에 매핑된다. 8-bytes access mode에서는 64bits/clock의 bandwidth를 가지므로 동일한 bank 내 2개의 32-bit words에 액세스하는 것이 항상 bank conflict를 발생시키지는 않는다. 한 클럭 내에 64-bits를 읽고, 요청된 32 bits만 각 스레드에 전달할 수 있다.

아래 그림은 32-bit(4-byte) mode에서 바이트 주소와 word index의 매핑을 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FsPnJL%2Fbtr1pXgjUZ5%2FsxFzUEWM5tLwuUUOrtIsoK%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

그리고 아래 그림은 4-byte work index가 86-bit(8-byte) mode bank에 어떻게 매핑되는지를 보여준다. Word 0과 word 32가 동일한 bank 0에 속하지만, 동일한 메모리 요청으로 이 두 word를 읽는 것이 bank conflict를 의미하지는 않는다.

아래 그림은 64-bit mode에서 conflict-free access의 한 예시를 보여준다. 여기서 각 스레드는 서로 다른 bank에 액세스한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbgwh5I%2Fbtr1dmIgyT6%2FMpV2lAUSm9GLtccWFdU1z1%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

아래 그림은 64-bit mode에서 또 다른 conflict-free acess 패턴을 보여준다. 두 스레드가 동일한 bank의 words에 액세스하고 있지만, 동일한 8-byte word에 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdfrpys%2Fbtr0RJSCHoW%2FDVXXaUlUoOGfbuUxPn7UVK%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

아래 그림은 two-way bank conflict를 보여준다. 두 스레드가 같은 bank에 액세스하고 있는데, 그 메모리 주소는 서로 다른 8-byte words에 속한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMDZKO%2Fbtr1lzNR1m8%2FjuHoC3NvvDzY71DIKvfVuk%2Fimg.png" width=400px style="display: block; margin: 0 auto"/>

아래 그림은 three-way bank conflict를 보여준다. 3개의 스레드가 같은 bank에 액세스하고, 그 메모리 주소는 서로 다른 8-byte words에 속하고 있다는 것을 알 수 있다.

## Memory Padding

**Memory padding**은 bank conflict를 피할 수 있는 한 가지 방법이다. 간단한 예시를 통해 이를 살펴보자. 먼저 shared memory bank가 32개가 아닌 5개로 구성되어 있다고 가정해보자. 만약 모든 스레드들이 bank 0의 서로 다른 위치를 액세스한다면, five-way bank conflict가 발생할 것이다. 이러한 bank conflict를 해결하는 한 가지 방법은 N번째 요소 이후마다 하나의 word padding을 추가하는 것이다. 이때 N의 값은 bank의 수가 된다. 이를 그림으로 나타내면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdHz9ka%2Fbtr0WfcBSQU%2FGWabcbKWU8tHippSh2hibK%2Fimg.png" width=500px style="display: block; margin: 0 auto"/>

Padding을 추가함으로써 동일한 bank에 위치했던 word를 서로 다른 bank에 위치시킬 수 있고, 결과적으로 서로 다른 bank로 액세스하게 하여 bank conflict를 해결할 수 있다.

패딩된 메모리는 사용되지 않는 dummy이며, 오직 데이터 요소들의 위치를 이동시켜 동일한 bank에 있던 요소들을 서로 다른 bank로 분산시키는 용도로 사용된다. 결과적으로 스레드 블록에서 사용할 수 있는 shared memory의 총량은 감소되며, 올바른 데이터 요소에 액세스하기 위해서 배열 인덱스도 다시 계산해야 한다.

## Access Mode Configuration

[Access Mode](#access-mode)에서 살짝 봤는데, 최근 GPU에서는 4-byte와 8-byte shared memory access mode를 지원한다. 기본 모드는 4-byte mode이다. 현재 적용된 access mode는 아래의 CUDA 런타임 API를 통해 쿼리할 수 있다.
```c++
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig);
```
결과는 `pConfig`에 리턴된다. 리턴되는 bank configuration은 아래의 두 가지 enum 값 중 하나이다.

- `cudaSharedMemBankSizeFourByte`
- `cudaSharedMemBankSizeEightByte`

아래의 CUDA 런타임 API를 사용하면 bank size를 설정할 수 있다.
```c++
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
```

사용 가능한 bank configuration은 다음과 같다.
- `cudaSharedMemSizeDefault`
- `cudaSharedMemBankSizeFourByte`
- `cudaSharedMemBankSizeEightByte`

커널 실행 사이에 shared memory configuration을 변경하려면 동기화 포인트가 필요할 수 있다. Shared memory bank size를 변경해도 shared memory 사용량이나 커널 점유율에 영향을 미치지는 않지만, 성능에 큰 영향을 미칠 수 있다. Bank size가 크면 shared memory access에 대해 더 높은 bandwidth를 달성할 수 있지만, shared memory access 패턴에 따라 더 많은 bank conflict가 발생할 수 있다.

<br>

# Configuring the Amount Of Shared Memory

RTX3080 기준, 각 SM에는 128KB의 on-chip이 있다. 이 on-chip memory는 shared memory와 L1 cache가 공용으로 사용하는데, CUDA는 L1 cache와 shared memory의 크기를 설정할 수 있는 두 가지 방법을 제공한다.

> [CUDA 문서](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x)에 의하면 compute capability 8.6에서 shared memory의 capacity는 0, 8, 16, 32, 64, or 100KB로 지정할 수 있다.

- Per-device configuration
- Per-kernel configuration

아래의 CUDA 런타임 API를 사용하면 커널이 사용할 L1 cache와 shared memory의 크기를 설정할 수 있다.

```c++
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
```

`cacheConfig` 인자는 현재 CUDA device에서 on-chip memory를 어떻게 L1 cache와 shared memory 분할할 것인지를 지정한다. 가능한 cache configuration은 다음과 같다.

- `cudaFuncCachePreferNone` : no preference for shared memory or L1 (default)
- `cudaFuncCachePreferShared` : prefer larger shared memory and smaller L1 cache
- `cudaFuncCachePreferL1` : prefer larger L1 cache and smaller shared memory
- `cudaFuncCachePreferEqual` : prefer equal size L1 cache and shared memory

> `cudaDeviceSetCacheConfig()`로 설정되는 옵션은 단지 선호할 뿐이다. 가능한 경우 요청된 config를 사용하지만, 필요한 경우 다른 config를 선택할 수 있다.

만약 커널 함수가 shared memory를 많이 사용하여 device에서 shared memory를 많이 사용하도록 설정하면 커널 점유율이 높아져 성능이 향상될 수 있다. 반면 커널이 shared memory를 많이 사용하지 않는 경우, L1 cache를 더 많이 사용하도록 구성하는 것이 좋다.

CUDA 런타임에서는 커널 함수마다 다른 cache config를 선택하도록 할 수 있다. 커널을 실행하기 전에 `cudaDeviceSetCacheConfig()`를 사용해 device cache configuration을 재정의할 수도 있지만, 아래의 런타임 API를 통해 커널 함수 별로 cache config를 설정할 수도 있다.
```c++
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
```

Config를 적용할 커널은 `func` 커널 함수 포인터로 지정된다. 각 커널에 대해 이 함수는 한 번만 호출하면 된다.

L1 cache와 shared memory는 동일한 on-chip 하드웨어에 위치하지만, 몇 가지 다른점이 있다. Shared memory는 32개의 bank를 통해 액세스되지만, L1 cache는 캐시 라인을 통해 액세스된다. 또한, shared memory를 사용하면 무엇이 어디에 저장되는지 완벽히 제어할 수 있지만, L1 cache는 하드웨어에 의해서 제어된다.

<br>

# Synchronization

병렬 스레드 간의 동기화는 모든 parallel computing language에서 핵심 메커니즘이다. 이름에서 알 수 있듯이 shared memory는 스레드 블록 내 여러 스레드에서 동시에 액세스할 수 있다. 그렇기 때문에 여러 스레드가 동기화없이 같은 shared memory 위치를 수정하려고 하면 **inter-thread conflict**가 발생하게 된다. CUDA는 intra-block synchronization을 위해 몇 가지 런타임 함수를 제공한다. 일반적으로 동기화에는 다음의 두 가지 기본 방식이 있다.

- Barriers
- Memory fences

Barrier에서는 모든 스레드들이 다른 모든 스레드가 이 barrier 지점에 도달할 때까지 기다린다. 반면 memory fence에서는 메모리에 대한 모든 수정이 다른 모든 스레드들에게 visible될 때까지 모든 스레드가 중지된다.

CUDA의 intra-block barriers와 memory fences를 살펴보기 전에 먼저 CUDA가 채택한 weakly-ordered memory model에 대해서 살펴보자.

## Weakly-Ordered Memory Model

현대 메모리 아키텍처는 relaxed memory model을 따른다. 이는 메모리 액세스가 프로그램 내에서 나타나는 순서대로 반드시 실행되는 것이 아니다. CUDA는 weakly-ordered memory model을 채택하여 더욱 공격적인 컴파일러 최적화를 가능하게 한다.

GPU 스레드가 shared memory, global memory, page-locked host memory 등 다른 메모리에 데이터를 write하는 순서는 소스 코드에 나타난 액세스 순서와 반드시 같을 필요가 없다. 한 스레드에서의 write가 다른 스레드들에게 visible되는 순서는 이 write가 실제 수행된 순서와 일치하지 않을 수 있다.

스레드가 다른 메모리로부터 데이터를 읽는 순서는 read instruction이 독립적인 경우, 프로그램에 나타나는 순서와 반드시 같을 필요가 없다.

프로그램의 정확성을 위해 특정 순서를 명시적으로 적용하려면 어플리케이션 코드에 memory fences나 barriers를 추가해주어야 한다. 이는 다른 스레드와 리소스를 공유하는 커널의 올바른 동작을 보장하는 유일한 방법이다.

## Explicit Barrier

CUDA에서 같은 스레드 블록 내 스레드들 사이에서만 barrier를 수행할 수 있다. 커널 함수 내에서 barrier point는 아래의 CUDA 내장 함수를 호출하여 지정할 수 있다.
```c++
void __syncthreads();
```

`__syncthreads()`는 블록 내 모든 스레드들이 이 지점에 도달할 때까지 모든 스레드들을 기다리도록 한다. 또한, 이 barrier 지점 이전에 발생한 모든 global, shared memory 액세스를 같은 블록 내 모든 스레드들에게 동기화되는 것을 보장한다.

`__syncthreads()`는 같은 블록 내 스레드들 간의 통신에도 사용된다. 블록 내 어떤 스레드들이 shared memory나 global memory의 같은 주소에 액세스할 때, 잠재적으로 발생할 수 있는 충돌(read-after-write, write-after-read, write-after-write)을 피할 수 있도록 해준다.

조건문 코드에서 `__syncthreads()`를 사용할 때는 특히 주의해야 한다. 오직 어떤 조건이 스레드 블록 내에서 모두 동일하게 평가되는 경우에만 `__synchthreads()`를 호출해야 한다. 그렇지 않으면 실행이 중단되거나 의도하지 않은 부작용이 발생할 수 있다. 예를 들어, 아래 코드에서는 스레드 블록의 모든 스레드들이 같은 barrier에 도달하지 않는다. 따라서, 스레드 블록 내 스레드들이 서로를 무한히 대기하게 된다.
```c++
if (threadID % 2 == 0) {
    __syncthreads();
}
else {
    __syncthreads();
}
```

블록 간의 동기화는 불가능하기 때문에 각 스레드 블록은 임의의 SM에서 임의의 순서로 실행될 수 있다 (compute capability 9.0에서 cluster 내의 thread block 간 동기화는 가능한 것으로 보임).

## Memory Fence

**Memory fence** 함수들은 fence 이전의 어떤 memory write가 fence 이후에 다른 모든 스레드들에게 동기화되도록 보장한다. 원하는 scope에 따라 block, grid, system의 3가지 memory fence를 제공한다.

아래 예제 코드에서 동일한 블록 내에 있는 thread 1은 `writeXY()`를 수행하고, thread 2는 `readXY()`를 수행한다고 가정해보자.
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

두 스레드는 동일한 메모리 위치인 `X`와 `Y`에 read/write를 동시에 수행한다. 모든 data-race는 undefined behavior을 발생시키므로, `A`와 `B`의 값은 어떠한 것도 될 수 있다.

다음의 내장 함수를 사용하면 스레드 블록 내에서 memory fence를 생성할 수 있다.
```c++
void __threadfence_block();
```

`__threadfence_block()`은 호출한 스레드 블록에 의한 shared memory나 global memory에 대한 모든 write가 fence 이후에 같은 블록 내 모든 스레드들에게 동기화되도록 보장한다. Memory fence는 스레드 동기화를 수행하는 것은 아니므로, 블록 내 모든 스레드가 이 명령어를 실제로 수행해야할 필요는 없다.

Grid 레벨에서 memory fence를 생성하려면 아래 내장 함수를 사용하면 된다.
```c++
void __threadfence();
```

`__threadfence()`는 global memory에 대한 write가 같은 grid 내의 모든 스레드들에 동기화될 때까지 모든 스레드들을 중지(stall)시킨다.

아래 내장 함수를 통해 system(including host and device) 간 memory fence를 생성할 수도 있다.
```c++
void __threadfence_system();
```
이 함수는 global memory, page-locked host memory, 다른 장치 memory에 대한 모든 write가 모든 device와 host threads에 동기화되도록 모든 스레드들을 중지시킨다.

## Volatile Qualifier

컴파일러는 global memory 또는 shared memory에 대한 read/write를 자유롭게 최적화할 수 있다 (memory fence나 synchronization을 해치지 않으면서). 예를 들어, global reads를 registers나 L1 cache에 캐싱하는 방식으로 최적화한다.

이러한 최적화는 **volatile qualifier**를 사용하여 비활성화할 수 있다. Global 또는 shared memory 변수가 `volatile` 키워드로 선언되면, 컴파일러는 해당 값이 다른 스레드에 의해 언제든지 변경되거나 사용될 수 있다고 가정한다. 따라서, 이 변수에 대한 모든 참조(reference)는 실제 memory read or write instruction으로 컴파일된다. 즉, 이 변수에 대한 모든 참조는 캐시를 스킵하게 된다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
- [NVIDIA CUDA Documentation: Memory Optimizations - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [NVIDIA CUDA Documentation: Compute Capability 5.x - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)
- [NVIDIA CUDA Documentation: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- [NVIDIA CUDA Documentation: Volatile Qualifier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier)