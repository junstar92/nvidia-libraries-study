# Table of Contents

- [Table of Contents](#table-of-contents)
- [Concurrent Kernel Execution](#concurrent-kernel-execution)
- [Concurrent Kernels in Non-NULL Streams](#concurrent-kernels-in-non-null-streams)
- [False Dependencies](#false-dependencies)
- [Concurrency-Limiting GPU Resources](#concurrency-limiting-gpu-resources)
- [Blocking Behavior of the Default Stream](#blocking-behavior-of-the-default-stream)
- [Create Inter-Stream Dependencies](#create-inter-stream-dependencies)
- [References](#references)

<br>

# Concurrent Kernel Execution

스트림에 대한 개념과 관련된 API 함수들은 [Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md)에서 자세히 다루었다. [Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md)에서 간단한 예제도 살펴봤지만, 이번 포스팅에서 조금 더 자세하고 더 많은 예제 코드를 통해 스트림을 사용하는 방법과 그 동작 방식에 대해서 자세히 살펴본다.

<br>

# Concurrent Kernels in Non-NULL Streams

스트림을 통한 동시 커널 실행을 시각적으로 확인하려면 `nsight systems`이 필요하다. 자세한 사용법을 숙지하고 있을 필요는 없으며, 필자의 경우 아래의 커맨드를 통해 생성되는 프로파일링 파일(`.nsys-rep`)을 `nsight systems`(GUI)로 열어서 확인한다.
```
$ nsys profile ./program
```

또는, GUI를 통해 프로파일링을 수행하고 바로 결과를 시각적으로 확인할 수 있는데, 사용법은 [Nsight Systems 문서](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gui-profiling)에서 확인할 수 있다. 사용법이 간단하므로, 여기서 따로 다루지는 않는다.

간단하게 구현된 4개의 커널과 4개의 스트림을 사용하여 각 스트림의 커널 실행이 동시에 수행되는 것을 테스트해보자. 커널 구현은 다음과 같으며, `kernel_2`, `kernel_3`, `kernel_4`도 모두 동일하게 구현되어 있다. 우리는 각 스트림에서 실행되는 4개의 커널들이 다른 스트림의 커널들과 동시에 실행되는 것을 `nsight systems`로 확인할 예정이다.

> 사용 중인 RTX3080의 경우, 성능이 좋아서 그런지 커널 간 오버랩되는 부분이 매우 적었다. 동시에 실행되지 않는 것은 아니지만, 하나의 커널이 다음 커널과 오버랩되기 전에 이미 실행을 완료해버린다. 시각적으로 커널들이 동시에 실행되는지 확실하게 확인하기 위해서 `printf` 출력을 통해 커널의 수행시간을 늘려서 테스트했다.

```c++
__global__
void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
        printf("%f\n", sum);
    }
}
```

동시 커널 실행을 테스트하기 위해, 먼저, non-null 스트림을 생성한다.
```c++
// allocate and initialize an array of stream handles
cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
}
```

그리고 커널들은 for loop를 통해 각 스트림에서 한 번씩 실행된다. 그리드와 블록의 크기는 모두 1로 지정했다.
```c++
for (int i = 0; i < num_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();
}
```

> 전체 코드는 [concurrent_exec.cu](/cuda/code/streams/concurrent_exec.cu)에서 확인할 수 있다.

전체 코드를 컴파일하고, 실행하면 아래와 같은 출력 결과를 얻을 수 있다. 스트림의 동작을 제대로 시각화하기 위해서 각 커널에서 `printf` 출력을 수행하는데, 아래 출력은 `printf`를 주석처리하고 실행한 출력 결과이다.
```
$ ./concurrent_exec
> At device 0: NVIDIA GeForce RTX 3080 with num_streams=4
> Compute Capability 8.6 hardware with 68 multi-processors
Measured time for parallel execution: 0.040 ms
```

`printf`로 출력하도록 하고, `nsight systems`로 프로파일링하면 아래의 결과 화면을 볼 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdWEH7X%2Fbtr18DIxJ3b%2FTHW861vxbS0G55WTa0pnLk%2Fimg.png" width=800px style="display: block; margin: 0 auto"/>

예상한 대로 각 스트림에서의 커널들의 동시에 실행되고 있는 것을 확인할 수 있다.

<br>

# False Dependencies

[Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md#false-dependencies)에서 **False Dependency** 에 대해서 언급했었다. False dependency는 Hyper-Q를 지원하지 않는 GPU에서 물리적인 hardware work queue가 하나뿐이기 때문에 발생하는 현상이다. 각 스트림에서 커널들을 각각 실행하더라도, host 측에서의 커널 실행 순서에 의해 스트림간 종속성이 발생하여 결과적으로는 종속성이 없는 부분에서만 동시에 커널이 수행되는 현상이며, 최근 대부분의 GPU에서는 Hyper-Q를 지원하기 때문에 크게 신경쓰지 않아도 되는 부분이다.

만약 Hyper-Q를 지원하지 않는다면, 위에서 살펴본 4개의 커널 실행은 아래와 같이 실행된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCmUy0%2Fbtr1XPpzjuW%2F6s5SKhS7p5hdyByDyjQND1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

위의 상황에서는 `kenel_1`, `kernel_2`, `kernel_3`, `kernel_4`는 아래의 순서대로 host 측에서 호출되는데, 빨간색 원으로 표시된 부분에서만 커널이 독립적으로 동시에 실행된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMMvr1%2Fbtr1Vdx0WNG%2F5VJzQF0hhry0dGEiVhXKYK%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

Hyper-Q가 없더라도 false dependency를 해결할 수는 있다. 바로 host 측에서 수행할 커널을 **breadth-first** order로 실행하면 된다. 기존에는 위 그림과 같이 for loop를 통해 하나의 스트림에서 4개의 커널을 순차적으로 실행하고, 다음 반복에서 다른 스트림에서 4개의 커널을 순차적으로 실행한다. 이는 커널들을 **depth-first** order로 실행한다. 반면 breath-first order에서는 첫 번째 커널을 4개의 스트림에서 실행시키고, 그 다음에 두 번째 커널을 다시 4개의 스트림에서 실행시키는 방식이다. 코드로 표현하자면 다음과 같다.
```c++
// dispath job with breadth-first order
for (int i = 0; i < num_streams; i++)
    kernel_1<<<grid, block, 0, streams[i]>>>();
for (int i = 0; i < num_streams; i++)
    kernel_2<<<grid, block, 0, streams[i]>>>();
for (int i = 0; i < num_streams; i++)
    kernel_3<<<grid, block, 0, streams[i]>>>();
for (int i = 0; i < num_streams; i++)
    kernel_4<<<grid, block, 0, streams[i]>>>();
```

따라서, breadth-first order로 실행하면 아래와 같이 커널들이 실행된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcPTIJP%2Fbtr1XzgcUUl%2FZgiWFnny11r4oG62XMASMk%2Fimg.png" width=600px style="display: block; margin: 0 auto"/>

그 결과, hardware work queue에서 인접한 커널들끼리는 서로 다른 스트림에 속하기 때문에 인접한 커널들 간 false dependency가 사라지며, 커널들이 동시에 실행될 수 있다.

> 위에서 언급했지만, 현재 대부분의 device에서는 Hyper-Q를 지원하므로 이 현상에 대해서 신경쓸 필요는 없다.

단, 요즘 GPU에서도 false dependency 현상이 발생하는 환경을 만들어줄 수는 있다. 바로 `CUDA_DEVICE_MAX_CONNECTIONS`라는 환경 변수의 값을 1로 설정하면 hardware work queue가 하나인 환경으로 설정된다. 즉, Hyper-Q에서 지원하는 hardware work queue의 갯수를 설정한다.

[concurrent_exec.cu](/cuda/code/streams/concurrent_exec.cu)에서 main문 시작 부근에 아래 코드를 주석 처리해두었는데,
```c++
// set up max connectioin (hyper-q)
char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
setenv (iname, "1", 1);
char *ivalue =  getenv (iname);
printf ("%s = %s\n", iname, ivalue);
```
이 주석을 해제하고 컴파일 및 `nsight systems`로 프로파일링해보면, 아래의 결과를 확인할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbfc74R%2Fbtr2gkoliQl%2FlXzKkJhaKktX00USOhEcz0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

> Linux bash에서 `export CUDA_DEVICE_MAX_CONNECTIONS=1`로 `CUDA_DEVICE_MAX_CONNECTIONS` 값을 지정할 수도 있는데, 필자의 경우 프로그램에서 환경변수가 적용되지 않았다 (이유를 모르겠다...).

Depth-first order의 커널 실행을 breadth-first order로 변경해서 다시 프로파일링해보면 각 스트림의 커널들이 동시에 실행되는 것을 확인할 수 있다.

<br>

# Concurrency-Limiting GPU Resources

GPU 리소스는 제한되어 있기 때문에 프로그램 내에서 동시에 실행할 수 있는 커널의 수는 제한되어 있다. 이전 예제 코드 [concurrent_exec.cu](/cuda/code/streams/concurrent_exec.cu)에서는 이러한 리소스 제약을 피하기 위해서 그리드와 블록의 사이즈를 모두 1로 지정하여 커널을 실행했다. 예제 코드에서 실행되는 커널들은 아주 적은 리소스만을 사용한다.

실제로 필드에서 사용되는 프로그램에서 커널들은 많은 스레드들로 실행된다. 일반적으로 수백 개 이상의 스레드가 생성되어 커널을 수행하는데, 스레드가 너무 많으면 커널을 실행하는데 하드웨어 리소스가 부족할 수 있다. 이러한 현상을 실제로 관찰해보려면 이전 예제 코드에서 더 많은 스레드를 사용하고 그리드에 하나 이상의 블록이 포함되도록 변경하여 테스트해볼 수 있다.

먼저 기존 코드 그대로 컴파일하고, 스트림의 갯수만 32개로 지정하여 프로파일링한 결과는 다음과 같다.
```
$ nsys profile ./concurrent_exec 32
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FPtMv6%2Fbtr2vnSFl5o%2FkvOAwOpYAGJIzjNDalQTBk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

스트림이 너무 많아서 하나의 이미지에 담지는 못했지만, 32개의 각 스트림에서 커널들이 모두 동시에 수행되는 것을 확인할 수 있다.

이제, [concurrent_exec.cu](/cuda/code/streams/concurrent_exec.cu) 코드에서 아래와 같이 수정해 컴파일하고, 프로파일링한다.
```c++
// concurrent_exec.cu
...
#define N 1
...
int main(int argc, char** argv)
{
    ...
    dim3 block(128);
    dim3 grid(32);
    ...
}
```

> 커널 내에서 `printf`를 수행하고 있으므로, 그리드 내 스레드 수를 위와 같이 늘려주면 프로그램의 실행시간이 매우 길어진다. 따라서, 테스트를 위해서 N을 1로 감소시켜 테스트했다. 

결과는 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FVFSQG%2Fbtr2uZ5xbbm%2Fltjl7dKYGLhqYW6SoZgl81%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

예상한대로 동시에 실행되는 커널의 수가 감소한 것을 볼 수 있다. 이는 아까 언급했듯이 모든 커널을 동시에 실행시키기에는 GPU의 리소스가 부족하기 때문이다.

<br>

# Blocking Behavior of the Default Stream

[concurrent_exec.cu](/cuda/code/streams/concurrent_exec.cu) 코드 내에서 커널을 실행하는 부분을 아래와 같이 변경시켜 보자. 아래 코드에서는 3번째 커널 실행(`kernel_3`)을 default 스트림으로 실행시킨다 (스트림의 수는 4로 지정됨).
```c++
for (int i = 0; i < num_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();
}
```

`kernel_3`은 default 스트림에서 실행되기 때문에 해당 커널 이후에 호출된 커널들은 `kernel_3`이 완료될 때까지 블로킹된다. 이를 `nsight system`으로 확인해보면 다음과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbfc74R%2Fbtr2gkoliQl%2FlXzKkJhaKktX00USOhEcz0%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

[Introducing CUDA Streams: Blocking and Non-blocking Streams](/cuda/study/14_introducing_cuda_streams.md#blocking-and-non-blocking-streams)에서 설명한 내용이므로 자세한 내용은 이를 참조 바람.

<br>

# Create Inter-Stream Dependencies

이상적으로는 스트림 간의 의도하지 않은 종속성은 없어야 한다. 하지만 복잡한 프로그램에서는 다른 스트림에서의 작업이 어떤 스트림이 수행되기 전에 완료될 필요가 있거나 그 결과가 필요할 때가 있다. 이러한 경우에는 스트림 간 종속성을 추가하는 것이 유용할 수 있다.

스트림 간의 종속성은 **CUDA Event** 를 사용하여 추가할 수 있다. CUDA 이벤트를 사용하는 방법과 이에 관련한 내용은 [Introducing CUDA Event](15_introducing_cuda_event.md#creating-inter-stream-dependencies)에서 다루도록 한다.


<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher
