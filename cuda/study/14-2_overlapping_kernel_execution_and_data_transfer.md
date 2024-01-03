# Table of Contents

- [Table of Contents](#table-of-contents)
- [Overlapping Kernel Execution and Data Transfer](#overlapping-kernel-execution-and-data-transfer)
- [Overlap Using Depth-First Scheduling](#overlap-using-depth-first-scheduling)
- [Overlap Using Breadth-First Scheduling](#overlap-using-breadth-first-scheduling)
- [References](#references)

<br>

# Overlapping Kernel Execution and Data Transfer

지난 포스팅인 [Introducing CUDA Stream](/cuda/study/14_introducing_cuda_streams.md)과 [Concurrent Kernel Execution](/cuda/study/14-1_concurrent_kernel_execution.md)에서 CUDA 스트림에 대한 기본 및 세부 사항들과 스트림을 사용하여 여러 커널들을 동시에 실행시키는 방법에 대해 살펴봤다. [Introducing CUDA Stream](/cuda/study/14_introducing_cuda_streams.md)에서 간단한 예제를 통해 살펴봤는데, 이번 포스팅에서는 kernel execution과 data transfer를 동시에 실행하는 방법에 대해서 살펴본다.

GPU에는 두 개의 **copy engine queue** 가 있다. 하나는 HtoD 방향으로의 전송용이고, 다른 하나는 DtoH 방향으로의 전송용이다. 따라서, 최대 2개의 data transfer를 오버랩할 수 있는데, 이는 두 data transfer의 방향이 서로 다르고, 서로 다른 스트림에서 수행되는 경우에만 가능하다. 이 조건이 만족되지 않으면 모든 data transfer는 순차적으로 수행된다.

또한, 아래의 두 가지 케이스 중, data transfer와 kernel execution 간의 관계가 어떠한지 알아야 한다.

- 커널 함수 내에서 data `A`를 소비한다면, `A`에 대한 data transfer는 동일한 스트림에서 커널이 실행되기 전에 준비되어야 한다.
- 만약 커널 함수가 `A`를 전혀 소비하지 않는다면, 커널과 데이터 전송은 서로 다른 스트림에 위치할 수 있다.

두 번째 경우, kernel과 data transfer가 동시에 실행될 수 있다는 것은 자명하다. 따라서, 단지 서로 다른 스트림에 속하도록 하는 것만으로 런타임에 동시에 실행해도 문제없다는 것을 알려준 것이다. 하지만, 커널이 데이터에 종속되는 첫 번째 경우에서는 data transfer와 kernel execution을 중첩시키는 것이 복잡하다. [Vector Addition](/cuda/study/14_introducing_cuda_streams.md#stream-example-vector-addition) 예제를 다시 살펴보면서 커널과 데이터 간 종속성이 존재할 때 어떻게 두 작업을 오버랩할 수 있는지 살펴보자.

> Vector Addition 예제의 전체 코드는 [vector_add_with_streams.cu](/cuda/code/streams/vector_add_with_streams.cu)를 참조

# Overlap Using Depth-First Scheduling

> CUDA 스트림에서 Depth-First Scheduling은 서로 다른 스트림의 작업을 하나씩 실행하는 것이 아닌, 한 스트림에 대한 작업을 모두 실행하고, 또 다른 스트림에 대한 작업을 그 다음에 모두 실행하는 방식을 의미한다. 즉, 하나의 for 문을 통해 한 스트림에 대한 작업을 모두 실행시키고, 다음 반복에서 다음 스트림에 대한 작업을 모두 실행시키는 것이다. Depth-First order에 대한 내용은 [False Dependencies](/cuda/study/14_introducing_cuda_streams.md#false-dependencies)에서 간단히 설명하고 있다.

예제에서 실행할 커널은 다음과 같다.
```c++
__global__
void vectorAddOnGPU(float const* a, float const* b, float* c, int const num_elements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elements) {
        for (int i = 0; i < N; i++) {
            c[idx] = a[idx] + b[idx];
        }
    }
}
```

이전 포스팅에서 사용한 커널 함수와는 아주 약간 차이점이 있는데, `nsight systems`에서 커널과 데이터 전송 간의 오버랩을 조금 더 쉽게 관찰하기 위해 커널의 수행 시간을 조금 늘려주려고 for문을 추가하였다. 그냥 동일한 연산을 `N`번 반복할 뿐이다.

CUDA 프로그램에서 vector addition은 기본적으로 아래의 구조로 구현된다.

1. 두 입력 벡터를 device로 복사
2. Vector addition 수행
3. 하나의 출력 벡터를 다시 host로 복사

위 과정만 따르면 당연히 연산과 데이터 전송을 오버랩하여 실행시킬 수 없다. Vector addition에서 연산과 데이터 전송을 동시에 실행하려면, vector addition이라는 하나의 문제를 여러 하위 문제로 분할해야 한다. 즉, input과 output을 여러 subsets으로 분할하고, 하나의 subset의 작업과 다른 subset의 작업을 서로 오버랩해야 한다. Vector addition의 하위 문제는 서로 독립적이므로, 각 하위 문제들은 서로 다른 CUDA 스트림에서 독립적으로 실행될 수 있으며, 이를 통해 커널과 데이터 전송을 동시에 실행할 수 있게 된다.

Data transfer를 kernel execution과 오버랩시키려면, 비동기 복사 함수인 `cudaMemcpyAsync`를 반드시 사용해야 한다. 하지만, `cudaMemcpyAsync`를 사용하려면 `pinned host memory`가 반드시 필요하다. 따라서, 비동기 복사를 위해 host memory를 `cudaMallocHost`를 사용하여 pinned memory로 할당해주어야 한다.
```c++
// allocate pinned host memory for asynchronous data transfer
cudaMallocHost(&h_a, bytes);
cudaMallocHost(&h_b, bytes);
```

예제 코드에서는 4개의 스트림을 사용하므로, 데이터를 균등하게 4등분한다.
```c++
size_t num_elements_per_stream = num_elements / NUM_STREAMS;
```

그런 다음, 반복문을 통해 각 스트림에서 `num_elements_per_stream` 개의 요소에 대해 data transfer와 kernel을 실행하도록 한다.
```c++
for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * num_elements_per_stream;
    cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
    vectorAddOnGPU<<<grid, block, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, num_elements_per_stream);
    cudaMemcpyAsync(gpu_ref + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);
}
```

Default 스트림으로 데이터 분할없이 vector addition을 수행한 것과 4개의 스트림으로 데이터를 4등분하여 비동기로 실행한 결과는 다음과 같다.
```
$ ./vector_add_streams
> Vector Addition + Data Transfer(HtoD, DtoH) at device 0: NVIDIA GeForce RTX 3080
> with 16777216 elements
vectorAddOnGPU with default stream         : elapsed time 31.859455 ms
vectorAddOnGPU with non-default streams(4) : elapsed time 7.565312 ms
```

실행 결과, 4개의 스트림으로 실행헀을 때가 default 스트림만으로 실행했을 때보다 약 4배 이상 빠르다.

그 이유는 `nsight systems`를 통해 살펴볼 수 있다.

아래 이미지는 default 스트림으로 데이터 분할없이 실행할 때의 시퀀스이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcXB0WP%2Fbtr2rFG1N9r%2Ffm5io9AzSKHN6rEeXXJK3k%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

그리고 아래 이미지는 4개의 스트림으로 실행할 떄의 시퀀스이다.
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHIjQ3%2Fbtr2sYzs2XH%2FZyViH98WoKkZ9H2afnvjS1%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

이 결과를 통해 3가지 타입의 오버랩을 관찰할 수 있다.

- 다른 스트림 간의 kernel execution (커널 수행 시간이 짧아 위 이미지에서는 관측되지 않음, [Concurrent Kernel Execution](/cuda/study/14-1_concurrent_kernel_execution.md)에서 확인 가능)
- 한 스트림의 kernel exceution과 다른 스트림의 data transfer 간 오버랩
- 서로 다른 스트림에서 방향이 다른 data transfers 간의 오버랩

위 그림은 스트림의 blocking 동작도 보여준다.

- 동일한 스트림 내에서 kernel execution은 이전에 실행된 data transfer에 의해 블로킹된다
- 한 스트림에서 HtoD data transfer는 이전에 실행된 (다른 스트림 또는 동일한 스트림에서의) 동일한 방향의 data transfer에 의해 블로킹된다

즉, HtoD data transfer는 4개의 서로 다른 스트림에서 실행되지만, 실제로는 동일한 copy engine queue를 통해 실행되므로 타임라인에서 순차적으로 실행되는 것을 확인할 수 있다.

<br>

# Overlap Using Breadth-First Scheduling

Breadth-First Scheduling 또한 [False Dependencies](/cuda/study/14_introducing_cuda_streams.md#false-dependencies)에서 설명했으므로, 자세한 설명은 생략한다.

Breadth-First order에서 스트림을 사용한 vector addition의 구현은 다음과 같다. 메모리 할당이나 다른 부분은 동일하며, data transfer와 kernel execution의 구현이 조금 달라진다.
```c++
// initiate all asynchronous transfers to the device
for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * num_elements_per_stream;
    cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
}
// launch a kernel in each stream
for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * num_elements_per_stream;
    vectorAddOnGPU<<<grid, block, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, num_elements_per_stream);
}
// queue asynchronous transfer from the device
for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * num_elements_per_stream;
    cudaMemcpyAsync(gpu_ref + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);
}
```

프로파일링 결과는 다음과 같다.
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpRdF7%2Fbtr2sE8VJzt%2FGLD4fnbrydzcqDQbdydS50%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

Depth-First order와 거의 동일한 결과가 나오는 것을 확인할 수 있다.

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher

