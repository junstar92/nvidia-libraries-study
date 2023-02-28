# Table of Contents

- [Table of Contents](#table-of-contents)
- [Matrix Addition with Unified Memory](#matrix-addition-with-unified-memory)
- [References](#references)

<br>

# Matrix Addition with Unified Memory

[Memory Management](/cuda-study/10_memory_management.md)에서 **Unified Memory**에 대해 살펴봤었다. Unified Memory를 사용하면 CPU와 GPU 간의 data migration을 명시적으로 해줄 필요가 없어서 간결한 코드 작성이 가능하다. 또한, host와 device 메모리를 위한 포인터를 각각 사용할 필요없이 하나의 통합된 포인터 주소로 host와 device에서 액세스할 수 있다. 이번 포스팅에서는 unfied memory를 사용하는 행렬 덧셈 예제를 통해 unified memory를 어떻게 사용하는지, 그리고 기존의 메모리 사용 방법과 어떠한 성능 차이가 있는지 살펴본다.

Unified Memory는 host 코드 측에서 아래와 같이 할당할 수 있다.
```c++
float *A, *B, *gpu_ref;
cudaMallocManaged(&A, bytes);
cudaMallocManaged(&B, bytes);
cudaMallocManaged(&gpu_ref, bytes);
```

그리고 이렇게 할당한 unified memory는 host 측 함수 `initMatrix`를 통해 초기화할 수 있다. `initMatrix` 함수는 host 코드이다.
```c++
initMatrix(A, nx * ny);
initMatrix(B, nx * ny);
```

그리고 `A`와 `B`를 커널 함수 `matrixAddGPU`로 전달하여 행렬 덧셈을 수행할 수 있다.
```c++
matrixAddGPU<<<grid, block>>>(A, B, gpu_ref, nx, ny);
cudaDeviceSynchronize();
```

커널 함수를 호출한 뒤, host 측에서 `cudaDeviceSynchronize()`를 호출해주고 있다. 커널 함수는 기본적으로 비동기 호출이다. 기존의 메모리 사용 방법이라면 `cudaMemcpy`를 통해 동기화를 내부적으로 수행하고 있었지만, unified memory를 사용하면 host와 device 간의 명시적인 메모리 동기화가 필요하다.

> 명시적인 memory transfer가 적용된 전체 코드는 [matrix_add_manual.cu](/code/cuda/matrix_add/matrix_add_manual.cu), unified memory를 사용한 전체 코드는 [matrix_add_managed.cu](/code/cuda/matrix_add/matrix_add_managed.cu)를 참조

[matrix_add_manual.cu](/code/cuda/matrix_add/matrix_add_manual.cu)와 [matrix_add_managed.cu](/code/cuda/matrix_add/matrix_add_managed.cu)를 각각 컴파일한 뒤, 먼저 `managed`를 실행해보면 아래와 같은 출력 결과를 얻을 수 있다.
```
$ ./managed 14

> Matrix Addition(Managed) at device 0: NVIDIA GeForce RTX 3080
> with matrix 16384 x 16384
initialization:          3.607525 sec
matrixAdd on host:       0.578211 sec
matrixAdd on gpu :       0.004670 sec <<<(512,512), (32,32)>>>
```

다음으로 명시적인 데이터 전송을 사용하는 기존 방법이 적용된 `manual`을 실행하면 아래의 출력 결과를 얻을 수 있다.
```
$ ./manual 14

> Matrix Addition(Manual) at device 0: NVIDIA GeForce RTX 3080
> with matrix 16384 x 16384
initialization:          3.742100 sec
matrixAdd on host:       0.352517 sec
matrixAdd on gpu :       0.004698 sec <<<(512,512), (32,32)>>>
```

결과를 보면, managed memory를 사용하는 커널의 성능이 host와 device 간의 명시적인 데이터 복사를 사용하는 것만큼 빠르다는 것을 보여준다.

커널의 성능이 거의 동일하게 측정된 이유는 사실 코드 내에서 커널 코드를 측정하기 전에 warming-up 커널을 실행시켜주었기 때문이다. 만약 warming-up 커널이 없다면 managed memory를 사용하는 커널은 더 느린 속도를 보여줄 것이다. 실제로 warming-up 커널을 제거하고 실행한 결과를 보면 약 0.5초로 약 100배 더 느린 속도를 보여준다. 이는 warming-up 커널을 통해 커널에서 사용할 managed memory에 대한 HtoD migration을 미리 수행하기 때문이다. 만약 warming-up 커널이 없다면 성능 측정 커널을 수행하기 전에 HtoD migration이 진행되고, migration 시간이 측정에 포함되어 더 시간이 오래 걸리는 것으로 측정되는 것이다.
```
$ ./managed 14
> Matrix Addition(Managed) at device 0: NVIDIA GeForce RTX 3080
> with matrix 16384 x 16384
initialization:          3.593732 sec
matrixAdd on host:       0.585224 sec
matrixAdd on gpu :       0.483863 sec <<<(512,512), (32,32)>>>

$ ./manual 14
> Matrix Addition(Manual) at device 0: NVIDIA GeForce RTX 3080
> with matrix 16384 x 16384
initialization:          3.634217 sec
matrixAdd on host:       0.343543 sec
matrixAdd on gpu :       0.004697 sec <<<(512,512), (32,32)>>>
```

Unified memory의 동작 메커니즘을 간단히 살펴보자. Unified memory가 할당될 때, 처음에는 어느 위치(CPU or GPU)에 상주할 지 모른다. 할당된 이후, 처음으로 이 메모리를 사용하려고 하면, 페이지 폴트(page fault)가 발생하게 된다. 페이지 폴트가 발생되는 시점에서 host나 device는 요청된 메모리를 일괄적으로 migration하게 된다. 우리 예제에서는 managed memory를 할당한 뒤 host 측에서 `initMatrix` 함수를 통해 초기화를 수행하므로 CPU로 migration하고, host 측에서 이 메모리를 사용한다.

> `initialization` 시간에는 초기 CPU로 migration하는 시간까지 포함되어 있다고 생각했는데, `nsight system`으로 프로파일링한 결과, 할당된 다음 처음 발생하는 메모리 요청에 대해서는 migration이 발생하지 않는 것으로 확인된다. 그 결과, `initialization`에 걸리는 시간은 managed memory를 사용하는지 여부에 상관없이 거의 유사하다.

[matrix_add_managed.cu](/code/cuda/matrix_add/matrix_add_managed.cu)에서는 다음의 순서로 managed memory를 사용하게 된다.

1. `initMatrix`, `memset` : on host
2. `matrixAddonHost` : on host
3. `warmup` : on gpu
4. `matrixAddGPU` : on gpu
5. `checkResult` : on host

따라서, 2번에서 HtoD data migration(A, B matrix), 3번에서 HtoD data migration(gpu_ref matrix), 그리고 5번에서 DtoH migration(gpu_ref matrix)가 발생하게 된다.

이는 `nsight system`으로 확인해볼 수 있다. `--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true` 옵션을 통해 unified memory에서 발생한 페이지 폴트 정보도 프로파일링해볼 수 있다.
```
$ nsys profile --stats=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true ./managed 14
...
[7/11] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ---------------------------------
     73.7      202,161,395  43,958   4,599.0   3,519.0     1,470    46,847      4,935.1  [CUDA Unified Memory memcpy HtoD]
     26.3       72,066,805   6,144  11,729.6   2,511.5     1,056   109,537     19,242.4  [CUDA Unified Memory memcpy DtoH]

[8/11] Executing 'gpumemsizesum' stats report

 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation            
 ----------  ------  --------  --------  --------  --------  -----------  ---------------------------------
  3,221.225  43,958     0.073     0.053     0.004     0.778        0.114  [CUDA Unified Memory memcpy HtoD]
  1,073.742   6,144     0.175     0.033     0.004     1.044        0.301  [CUDA Unified Memory memcpy DtoH]
...
[10/11] Executing 'unifiedmemorytotals' stats report

 Total HtoD Migration Size (MB)  Total DtoH Migration Size (MB)  Total CPU Page Faults  Total GPU PageFaults  Minimum Virtual Address  Maximum Virtual Address
 ------------------------------  ------------------------------  ---------------------  --------------------  -----------------------  -----------------------
                      3,050.439                       1,073.676                 15,360                38,007  0x7F9C80000000           0x7F9D7FF83000
...
```

[8/11] 결과를 보면, 3개의 행렬(A, B, gpu_ref)가 HtoD로 memcpy 되었다는 것을 확인할 수 있다 (행렬 하나의 크기가 `16384 x 16384 x 4B = 약 1073 MB` 이다).

`nsight system`을 통해 생성된 결과 파일 `xxx.nsys-rep`를 `nsys-ui`를 통해 열어보면, 시각화된 정보로 확인할 수도 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8XcmJ%2Fbtr06WQDwWF%2FU3rrR8tNxAIBXxFk7wQbtk%2Fimg.png" width=700px style="display: block; margin: 0 auto"/>

<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher