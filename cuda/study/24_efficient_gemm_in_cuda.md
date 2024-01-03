# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Hierarchical Structure of Matrix Multiplication](#hierarchical-structure-of-matrix-multiplication)
  - [Threadblock-level GEMM](#threadblock-level-gemm)
  - [Warp-level GEMM](#warp-level-gemm)
  - [Thread-level GEMM](#thread-level-gemm)
- [Epilogue](#epilogue)
- [Optimizations](#optimizations)
  - [Pipelining](#pipelining)
  - [Threadblock Rasterization](#threadblock-rasterization)
  - [Parallelized Reductions](#parallelized-reductions)
- [References](#references)

<br>

# Introduction

NVIDIA 오픈소스인 [CUDA Templates for Linear Algebra (CUTLASS)](https://github.com/NVIDIA/cutlass/tree/main)는 선형대수 서브루틴을 구현한다. 생각보다 설명도 잘 되어 있고, 무엇보다 계층 구조를 코드로 잘 표현하고 있다. 또한, C++ 템플릿을 사용하여 CUDA 코드를 작성하는 방법에 많은 도움을 주는 것 같다. 이번 포스팅에서는 CUTLASS에서 행렬 곱셈을 CUDA로 어떻게 구현하고 있으며, 어떻게 해야 성능이 좋은 구현이 가능한지 살펴볼 예정이다.

> 이 내용을 살펴보기 전에 먼저 shared memory를 사용하여 행렬 곱셈을 구현하는 것에 대해 이해를 하고 있으면, 조금 더 이해하기 수월하다. 또한, 행렬 곱셈 최적화에 대해서 다룬 이전 포스팅([link](/cuda/study/23_optimizing_a_matmul_kernel.md))에서는 코드와 함께 설명하고 있으므로 이를 먼저 살펴보고 보는 것을 추천한다.

# Hierarchical Structure of Matrix Multiplication

기본적으로 3중 루프의 중첩으로 계산되는 행렬 곱셈은 NVIDIA GPU 하드웨어, memory locality, CUDA parallel programming model의 동시성과 일치시키기 위해서 블록화되고 타일화될 수 있다. CUTLASS에서는 아래와 같이 표현되는 루프 중첩으로 표현되는 구조를 NVIDIA GPU에 매핑하여 GEMM을 구현한다.

```c++
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            //
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```
위와 같은 tiled loop nest는 **threadblocks**, **warps**, 그리고, **CUDA & Tensor Cores** 간의 동시성을 목표로 한다. 또한, 이 구조를 통해 shared memory와 register 내에서 memory locality의 이점을 활용한다.

아래 그림은 위 구조에서 데이터의 흐름을 보여주고 있으며, 이 구조가 바로 CUTLASS에서 구현된 hierarchical GEMM computation 이다. 각 단계에서는 CUDA execution model 내 동시성 계층과 메모리 계층의 수준에 해당하는 중첩된 타일링 수준을 나타내며, 왼쪽에서 오른쪽으로 갈수록 점점 더 세밀해진다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FM4mWw%2FbtszGNrGQDK%2F3HNzD6p4DTUKOUlP25mH7K%2Fimg.png" height=250px style="display: block; margin: 0 auto; background-color:white"/>

## Threadblock-level GEMM

각 스레드 블록은 입력 행렬의 타일을 반복적으로 로드하고 누적된 행렬 곱셈을 계산하여 output GEMM의 일부분을 계산한다. Threadblock 수준에서는 필요한 데이터가 global memory에서 로드된다. 일반적으로 blocking strategy는 효율성을 달성하는데 중요하지만, 여러 상충되는 목표와 균형을 맞추어야 한다. 예를 들어, 스레드 블록이 클수록 메모리로부터 fetch가 적어지므로 DRAM bandwidth가 병목을 일으키지 않는다. 하지만 스레드 블록이 크면 풀려는 행렬의 크기와 잘 일치하지 않을 수 있다.

GEMM의 `M` 또는 `N` 차원의 크기가 작은 경우, 스레드 블록이 부분적으로 풀려는 문제 크기의 범위를 벗어날 수 있으므로 스레드 블록 내의 일부 스레드는 의미있는 연산을 수행하지 못할 수도 있다. `M`과 `N`이 모두 작고, `K`가 큰 경우에서는 상대적으로 적은 수의 스레드 블록을 사용하고 GPU 내의 모든 멀티프로세서를 최대로 활용하지 못할 수도 있다. 이와 같은 경우에는 [Parallelized Reductions](#parallelized-reductions)에서 설명하는 것과 같이 GEMM의 `K` 차원을 여러 스레드 블록 또는 여러 워프(warp)로 분할하여 해결할 수 있다.

이 단계에서 global memory로부터 로드한 데이터는 shared memory에 저장하게 된다. 이렇게 shared memory에 저장된 데이터는 하위 계층 구조에서 사용하게 된다.

CUTLASS에서 CUDA Core를 사용하는 버전의 threadblock 구현은 [mma_pipelined.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_pipelined.h)에 있다. 스레드블록 레벨에서의 행렬 곱셈 구현은 아래와 같다. 크게 `prologue`와 `gemm_iters`로 나누어져 있는데, 이 구현은 pipelining을 적용한 것이며 일단 pipeline은 무시해도 무방하다.

> `prologue`를 사용하는 이유는 [Pipelining](#pipelining)에서 설명하고 있다. 이를 통해 _software pipelining_ 을 구현하고 스레드 내에서 computation과 memory accesses를 오버랩하여 latency hiding하는 것이다.
> 
```c++
/// Perform a threadblock-scoped matrix multiply-accumulate
CUTLASS_DEVICE
void operator()(
  int gemm_k_iterations,                            ///< number of iterations of the mainloop
  FragmentC &accum,                                 ///< destination accumulator tile
  IteratorA iterator_A,                             ///< iterator over A operand in global memory
  IteratorB iterator_B,                             ///< iterator over B operand in global memory
  FragmentC const &src_accum)                       ///< source accumulator tile
{
  // Prologue
  prologue(iterator_A, iterator_B, gemm_k_iterations);
  // Wait until we have at least one completed global fetch stage
  gmem_wait();
  // Perform accumulation in the 'd' output operand
  accum = src_accum;
  // Perform the MAC-iterations
  gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
}
```
이를 시각적으로 표현하면 다음과 같다. `gemm_iters()` 함수 내에서 `K` 차원을 따라서 반복하며 MMA를 수행하게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNftCQ%2FbtszekXex1B%2Fo2OGzvJgsFc4YzgE6qQ2Fk%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

Output 행렬에서 초록색으로 표현된 타일을 스레드블록 하나가 처리하게 되는 것이다. 이때, 입력 행렬에서 사용되는 데이터를 한 번에 shared memory로 저장할 수 없기 때문에(용량 부족) `K` 차원의 축을 따라서 반복하면서 행렬 곱셈을 계산하고 이를 누적하게 된다.

CUTLASS의 스레드블록 구현에서는 아래의 두 단계로 나누어져 있다.

- `prologue`
- `gemm_iter`

우선 두 단계를 구분할 필요없이 일반적인 shared memory를 사용하여 행렬 곱셈을 구현한 커널([smem_gemm_kernel](https://github.com/junstar92/nvidia-libraries-study/blob/main/cuda/code/perf_sgemm/src/kernel/src/02_smem_gemm.cu))을 생각하면 된다. 즉, K차원으로 iteration을 수행하면서 행렬 A와 B의 타일들에 대한 행렬 곱셈을 수행하고 이를 누적하여 최종적으로 output 행렬의 일부를 계산하는 것이다.


CUTLASS에서는 각 스레드 블록 내에서 행렬 A와 B에 대해 각 스레드가 로드하는 데이터를 매핑하기 위한 thread map과 이를 shared memory에 저장하기 위한 shared memory용 thread map을 구현하고 있다. 구현을 살펴본 결과, 행렬 A와 B가 row-major든 column-major든 간에 무조건 행렬 A의 요소에 대한 shared memory layout은 column-major이며, 행렬 B의 요소에 대한 shared memory layout은 row-major이다. 여기서 행렬 A의 타일에 대한 shared memory이 layout이 column-major라는 것은 `CtaTileM x CtaTileK` 크기의 타일을 shared memory로 전달할 때, 전치하여 `CtaTileK x CtaTileM`의 형태로 전달한다는 것을 의미한다.

이렇게 매핑하는 이유는 MMA를 수행하기 위해서 shared memory에서 각 스레드의 register로 데이터를 로드할 때, bank conflict를 최소화하기 위한 의도로 보인다. 이에 대한 내용은 더 하위 계층 구조를 살펴보면서 각 스레드가 어떤 요소들을 담당하게 되는지 살펴보면 더 쉽게 이해할 수 있다. 참고로 CUDA Core를 타겟으로하는 기본 threadblock 구현은 [default_mma_core_simt.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_simt.h)에서 확인할 수 있다.

일반적으로 사용되는 스레드블록 타일 하나의 크기와 하나의 스레드블록에서 스레드 갯수는 다음과 같다 (`CtaTileM x CtaTileN x CtaTileK`).

- `64 x 64 x 8` (64 threads)
- `128 x 128 x 8` (256 threads)

### Loading A and B then Storing to Shared

각 스레드블록에서 A와 B 행렬의 입력 데이터를 global memory로부터 읽고 shared memory로 저장하는 것에 대해 조금 더 자세히 살펴보자.

스레드블록 타일의 크기를 `128 x 128 x 8`이라고 한다면, 이 스레드블록 타일은 `128 x 8` 크기의 행렬 A와 `8 x 128` 크기의 행렬 B의 행렬 곱셈을 반복하면서 누적하게 된다. 그렇다면, 한 번의 루프에서 행렬 A에 대해서 `128 x 8`개의 요소를 global memory로부터 shared memory로 전달하고, 행렬 B에 대해서도 `8 x 128`개의 요소를 global memory로부터 shared memory로 전달하게 된다.

먼저 행렬 A에 대해서 global memory로부터 shared memory로 데이터를 이동시키는 것을 살펴보자. 결론부터 이야기하면 아래 그림과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZD3h6%2FbtszC0r6XDt%2FmvB90qwucfKtu50bdish00%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

일단 하나의 스레드가 한 번에 하나의 요소를 복사한다. 그리고 각 스레드는 한 번에 하나의 요소를 복사한다. 행렬 A가 row-major이므로 global memory에 대해서 colesced access pattern을 만족하려면 데이터를 읽을 때 그림과 같이 스레드가 배치되어야 한다. 그래야 인접한 스레드가 최대한 인접한 메모리를 읽게 된다. 이렇게 읽은 값은 shared memory에 전치되어 저장된다. Shared memory에 저장할 때는 인접한 메모리에 액세스하는 패턴은 신경쓰지 않아도 되며, bank conflict만 신경써주면 된다. 사실 위와 같이 shared memory에 스레드를 배치하면 bank conflict는 발생하기 쉽다. `CtaTileM`의 크기가 `128`이라고 가정했기 때문에 shared memory layout에서 각 열은 모두 동일한 bank에 속해 있을 것이다. 이때, 0~7 스레드가 동일한 열, 즉, 동일한 bank의 서로 다른 주소에 액세스하기 때문에 8-way bank conflict이 발생하게 된다. 8~15 스레드, 16~23 스레드, 24~31 스레드도 마찬가지이다. 따라서, shared memory에 padding을 추가해주거나 permutation을 통해 bank conflict가 발생하지 않도록 저장해주는 방식이 필요하다. Padding을 사용한다면, 이 경우에는 `8 x 128` 형태의 shared memory이므로 4개의 요소를 padding으로 각 행에 추가해주면 bank conflict를 해결할 수 있다.

다음으로 행렬 B에 대해서 global memory로부터 shared memory로 데이터를 이동시킬 때의 액세스 패턴을 살펴보자. 행렬 B의 경우에는 global memory와 shared memory의 액세스 패턴이 동일하다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGs5eD%2FbtszH47xkoj%2FiCBU1hzmWhKrXM6YzxACwK%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

각 스레드가 연속된 shared memory의 주소를 액세스하기 때문에 bank conflict가 발생하지 않는다. 따라서, bank conflict를 없애기 위한 별도의 작업이 필요없다.

## Warp-level GEMM

Warp-level GEMM은 CUDA execution model 내에서 warp-level의 parallelism을 구현한다. 한 스레드블록 내에서 여러 워프들은 shared memory의 데이터를 registers로 로드하고 계산한다. Warp-level GEMM은 `mma.sync` 또는 `wmma` instructions를 사용하여 Tensor Cores를 사용하도록 구현될 수도 있고, CUDA Cores를 사용하도록 thread-level matrix 연산으로 구현될 수도 있다. 이 포스팅에서는 thread-level GEMM도 살펴볼 예정이라서 CUDA Cores를 사용하는 것을 기준으로 설명한다.

Warp-level GEMM에서는 shared memory로부터 데이터를 읽고 registers에 저장하기 때문에 최대의 성능을 얻으려면 shared memory에 액세스할 때 bank conflict가 발생하지 않도록 해야 한다. 또한, 워프 내에서 데이터 재사용을 극대화하려면 큰 warp-level GEMM 타일이 선택되어야 한다. 아래 하나의 워프 타일에서 계산을 담당하는 행렬 A와 B의 요소 위치를 살펴보면 shared memory 내에서 인접한 주소에 위치하고 있음을 알 수 있다 ([Threadblock-level GEMM](#threadblock-level-gemm)에서의 shared memory layout 그림 참조). 따라서, bank conflict가 거의 발생하지 않는다. 아래 그림에서 각 스레드가 어떤 요소를 계산하는지 살펴보면 shared memory에서 어느 위치에 액세스하여 register로 데이터를 로드하는지 알 수 있고, 이를 통해 bank conflict가 발생하는지 여부를 확인해볼 수 있다.

하나의 스레드블록은 아래 그림과 같이 여러 개의 워프로 구성된다. 아래 그림에서 초록색으로 표시된 타일 하나에 대한 계산을 워프 하나가 담당하게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FczO2Un%2Fbtsy9OS7Zq3%2FP0SskkcQ6rowVDoumbk2d0%2Fimg.png" height=500px style="display: block; margin: 0 auto; background-color:white"/>

일반적으로 워프 타일의 크기는 `64 x 32 x 8`을 사용한다. 그리고, 위 그림은 스레드블록 타일의 크기가 `128 x 128 x 8`일 때를 표현한 것이다. 따라서, 하나의 스레드블록 타일에는 `2 x 4`개의 워프 타일이 존재하게 된다.

하나의 워프 타일을 자세히 살펴보자.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8glSi%2FbtszbgaI3Lt%2FHpU8ktljS2Fhq3wi4886c1%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

위 그림에서 오른쪽 그림이 바로 하나의 워프 타일을 나타낸 것이다. 여기서 작은 사각형 하나가 바로 스레드 하나가 담당하는 계산 영역이다. 편의상 스레드 타일이라고 칭하도록 하자. 워프 내 스레드 갯수는 총 32개인데, 스레드 타일의 갯수는 32개가 훨씬 넘는다. 이는 하나의 스레드가 여러 개의 스레드 타일을 담당하면 해결되는 문제이다. 워프 내 스레드들에 0번부터 31번까지 식별 번호를 부여했을 때, 각 스레드가 담당하는 스레드 타일의 위치는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FddCwd9%2Fbtsy8unf5I0%2FcvM9kgkjE3m1MGBBwoDLOk%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

결과적으로 하나의 스레드는 4개의 스레드 타일의 계산을 담당하게 된다. 

## Thread-level GEMM

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlAvUg%2FbtszCFuVXd8%2FdHb5rSK8fMB20Bkny5P93K%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

Blocking 전략에서 가장 낮은 수준이며, 각 스레드는 특정 갯수의 요소에 대한 처리를 담당하게 된다. 위의 예시의 경우에는 하나의 스레드가 `8 x 8` 크기의 행렬 계산을 담당한다. 스레드 내 레지스터는 해당 스레드에서만 액세스할 수 있으므로, math instructions에서 register의 값을 재사용할 수 있도록 구조화해야 한다. 그 결과 스레드 내에서 2D 형태로 구조화되고, 각 스레드는 CUDA Core에 대한 일련의 독립적인 math instructions를 issue하고 외적(outer product)를 계산하여 누적하게 된다.

`128 x 128 x 8` 스레드블록 타일, `64 x 32 x 8` 워프 타일 예시에서 각 스레드 블록은 아래 그림과 같이 `4 x 4` 크기의 스레드 타일을 계산하게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFhWqx%2FbtszcIqJBxA%2FW8vApJmKoa5j5p7U5Kw4ok%2Fimg.png" height=400px style="display: block; margin: 0 auto; background-color:white"/>

여기서 알 수 있듯이 행렬 A와 B로부터 각각 8개의 요소를 register로 로드하고, 이를 사용하여 `8 x 8` 크기의 외적(outer product)를 계산하여 누적한다.

> 위의 구조로 구현한 커널은 [06_warptiling_gemm.cu](/cuda/code/perf_sgemm/src/kernel/src/06_warptiling_gemm.cu)에서 확인할 수 있다.

# Epilogue

방금까지 설명한 내용은 `C = AB` 형태의 행렬 곱셈에 중점을 두었다. 계산 결과는 스레드 블록 내 각 스레드의 register에 저장되어 있다. 마지막으로 이 값을 행렬 C에 저장해야 하며, 행렬 C는 global memory에 상주한다.

Epilogue는 global memory로 최종 결과를 전달하기 전에 추가 작업할 수 있는 단계이다.

예를 들면, 각 스레드가 계산한 결과가 있는 register에서 행렬 C의 global memory로 결과값을 전달할 때, 필연적으로 각 스레드가 연속된 global memory에 액세스할 수 없다. 알다시피 global memory의 bandwidth를 최대한 활용하려면 colesced access 패턴을 만족시켜야 한다. 따라서, 이러한 액세스 패턴이 가능하도록 register에서 shared memory로 결과값을 전달한 뒤, colesced access 패턴으로 global memory로 다시 전달할 수 있도록 하는 작업이 포함될 수 있다.

다른 예시로는 행렬 곱셈 결과를 입력으로 사용하는 linear scaling 및 ReLU 등의 elementwise operations가 있다 (**kernel fusion**). 행렬 곱셈 최종 결과를 global memory로 저장한 다음, 다시 이를 이용해 elementwise operations를 수행한다고 생각해보자. 그렇다면 요소의 수만큼 global memory access가 발생하고, 이는 전체 연산에서 큰 로드를 차지하게 된다. 이러한 계산을 epilogue 단계에서 적용하면 register에 위치한 결과값에 추가적인 계산을 하게되어, 불필요한 global memory access를 제거할 수 있게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmcKBm%2FbtszLqpmAMk%2FVccaanxN6KJSesKHE3DRcK%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

아래와 같은 연산들을 epilogue 단계에서 추가로 작업할 수 있다.

- Elementwise Operators: scaling, bias, activation functions
- Data Type Conversion: FP32 -> FP16, INT32 -> INT8 등
- Matrix Update Operations: reductions across thread blocks ([Split K](#split-k---reduction-across-threadblocks) 참조)

# Optimizations

위에서 언급한 계층 구조는 NVIDIA GPU의 CUDA execution model과 CUDA/Tensor Cores에 대한 효율적인 매핑을 제공한다. 이번에는 추가적으로 설계 부분에서 최고의 성능을 얻기 위해 병렬성을 극대화하고 가능한 data locality를 활용하는 방법에 대해 언급한다.

## Pipelining

위에서 설명한 블록 구조에서는 각 스레드에서 많은 레지스터를 사용한다. 특히 accumulator 요소는 스레드 내 전체 레지스터의 절반 이상을 차지한다. 그 결과, 동시에 실행되는 threads/warps/threadblocks의 수가 상대적으로 낮다. 즉, 점유율이 비교적 낮다. 이 때문에 SM 내에서 context switch를 통한 memory latency 및 stalls를 hiding하는 GPU 기능이 제한된다.

CUTLASS에서는 memory latency에 의한 영향을 완화하기 위해 software _pipelining_ 을 사용하여 스레드 내 연산과 memory access를 중첩하는데, 아래 범위에서 double buffering을 사용하여 이를 구현한다.

- **Threadblock-scoped shared memory tiles** : Shared memory에 두 개의 타일을 할당한다. 하나는 현재 행렬 연산을 위한 데이터를 로드하고, 다른 하나는 다음 mainloop 반복을 위해 global memory로부터 로드한 데이터를 버퍼하는데 사용된다.
- **Warp-scoped matrix fragments** : Register에 두 개의 fragments를 할당한다. 하나는 현재 행렬 연산 동안 CUDA와 Tensor Cores에 전달하며, 다른 하나는 다음 warp-level 행렬 연산에 대한 share dmemory fetch returns를 받는데 사용된다.

> [Threadblock-level GEMM](#threadblock-level-gemm)에서 살펴봤던 `prologue()` 함수가 바로 이 파이프라인을 위해서 존재하는 것이다.

아래 그림은 CUTLASS GEMM에서 사용되는 pipelined mainloop body를 보여준다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F72GD0%2FbtszHumPr9q%2Fwo86l6FCED9zPm8Pn3LUH0%2Fimg.png" height=350px style="display: block; margin: 0 auto; background-color:white"/>

## Threadblock Rasterization

Cache level에서의 데이터 재사용을 극대화하기 위해서 CUTLASS에서는 GEMM의 logical partition에 대한 스레드블록 매핑에 영향을 주는 여러 함수들을 정의한다. 주로 L2 cache의 hit-rate을 증가시키기 위한 목적으로 보이며, 이에 대한 내용은 [Threadblock Swizzling](/cuda/study/27_threadblock_swizzling.md)에서 추가로 다루고 있다.

CUTLASS에서 이 기능은 [cutlass/gemm/threadblock_swizzle.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h)에 정의되어 있다.

## Parallelized Reductions

### Split K - reduction across threadblocks

행렬 곱셈 연산은 O(MN)의 독립적인 내적(inner product) 계산 간의 병렬성을 노출한다. GEMM의 문제 크기가 충분히 큰 경우에는 이론상 최대 연산 처리량에 근접할 수 있지만, 작은 경우에는 전체 GPU를 효율적으로 점유하기에 threadblock의 크기가 너무 작다.

문제의 크기가 작은 경우, 내적 계산 동안 수행되는 reduction을 병렬화하면 더 많은 스레드 블록을 동시에 실행하는 대규모 threadblock-level GEMM 타일에서 얻을 수 있는 처리량의 이점을 활용할 수 있다.

CUTLASS에서는 GEMM의 K 차원을 파티셔닝하여 각 파티션 별로 추가적인 스레드블록을 실행한다. CUTLASS에서는 이를 **parallel reduction splitK** 라고 부른다. 이 전략에서는 `partitionedK GEMM`과 `batched reduction`이라는 두 개의 커널이 필요하다.

`PatritionedK GEMM`은 `batched strided GEMM`의 한 종류와 유사하다. 예를 들어, 문제의 크기가 `m=128`, `n=128`, `k=4096`이고, `partition=16`인 경우를 생각해보자. 그 결과, 각 batch가 `m=128`, `n=128`, `k=256`인 16개의 `batched strided GEMMs`를 만든다. `k`의 값이 `partition`으로 나누어 떨어지지 않는 경우도 허용한다. 예를 들어, `m=128`, `n=128`, `k=4096`, `partition=20`인 경우, 처음 19개의 batch는 `m=128`, `n=128`, `k=4096/20 = 204`이고, 나머지 batch는 `m=128`, `n=128`, `k=220`이 된다.

`batched reduction` 커널은 `partitionedK GEMM`의 출력을 입력으로 사용하여 K 차원을 따라 reduction을 수행한다.

### Sliced K - reduction across warps

Split K의 시나리오와 비슷하게, sliced-k도 `M`과 `N` 차원의 크기는 작지만 `K` 차원의 크기는 큰 커널의 효율성을 향상시키는 것을 목표로 한다. Threadblock-level에서 `CtaTileN`과 `CtaTileM`의 값은 워프 간 작업 분할을 통해 병렬성을 노출한다. 워프 타일의 크기가 클수록 instruction-level parallelism (ILP) 및 재사용율이 향상되지만, 스레드블록 당 실행되는 워프의 수가 제한되어 효율성이 떨어진다.

이러한 경우, 효율성을 향상시키기 위해 `CtaTileK`를 따라서 워프 타일을 분할하면 CTA에서 더 많은 워프가 동시에 실행될 수 있도록 하여 하드웨어를 보다 효율적으로 사용하는데 도움이 된다. Slice-K 커널은 `CtaTileN`과 `CtaTileM` 차원뿐만 아니라 `CtaTileK` 차원에서도 참여하는 워프 사이에서 스레드블록의 계산을 세분화한다. 따라서, 마지막에는 참여하는 워프 간에 작은 reduction cost가 수반된다.

<br>

# References

- [CUTLASS Documentation: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [GTC 2018: CUTLASS - CUDA Template Library for Dense Linear Algebra at All Levels and Scales](https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)
- [SGEMM](https://github.com/NervanaSystems/maxas/wiki/SGEMM)
- [Register Cache: Catching for Warp-Centric CUDA Programs](https://developer.nvidia.com/blog/register-cache-warp-cuda/)
- [Nvidia Tensor Core-CUDA HGEMM Advanced Optimization](https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85)
- [A Generalized Micro-kernel Abstraction for GPU Linear Algebra](https://www.cs.utexas.edu/users/flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf)
- [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/pdf/2006.12645.pdf)