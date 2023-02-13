# Table of Contents

- [Table of Contents](#table-of-contents)
- [Device Memory](#device-memory)
- [References](#references)

<br>

# Device Memory

[Heterogeneous Programming](/cuda-doc/01_programming_guide/02_programming_model.md#heterogeneous-programming)에서 언급했듯, CUDA 프로그래밍 모델은 host와 device로 구성된 시스템이라고 가정하고, host와 device는 저마다의 분리된 메모리를 가지고 있다고 가정한다. 커널 함수는 device memory에서 동작하기 때문에 CUDA runtime은 device memory를 할당/해제하고, device memory를 복사하고, host memory와 device memory 간에 데이터를 전송하는 함수를 제공한다.

Device memory는 **linear memory** 또는 **CUDA arrays** 로 할당될 수 있다. CUDA arrays는 texture fetching에 최적화된 opaque memory layout이며, 공식 문서의 [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)에서 이에 대해 설명한다.

Linear memory는 single unified address space에 할당된다고 공식 문서에서 언급하고 있는데, 이는 CPU(host)와 GPU(device)가 모두 동일한 메모리를 포인터를 통해 액세스할 수 있다고 한다. 여기서 언급하는 바는 일반적으로 CPU와 GPU는 분할된 메모리 공간을 가지고 있지만, CUDA에서는 single unified address space를 제공하여 host와 device 간 명시적인 데이터 전송을 할 필요없이 하나의 포인터로 동작할 수 있다는 것을 의미하는 것 같다. Address space의 크기는 host system과 GPU의 compute capability에 따라 다르다.

||x86_64 (AMD64)|POWER (ppc64le)|ARM64|
|--|--|--|--|
|up to compute capability 5.3 (Maxwell)| 40bit | 40bit | 40bit |
|compute capability 6.0 (Pascal) or newer| up to 47bit | up to 49bit | up to 48bit|

<br>

Linear memory는 일반적으로 `cudaMalloc()`을 사용하여 할당되고, `cudaFree()`를 사용하여 할당된 메모리를 해제한다. 그리고 host memory와 device memory 간의 data transfer는 일반적으로 `cudaMemcpy()`를 사용한다. [vector_add.cu](/code/cuda/vector_add/vector_add.cu)를 보면, device memory의 할당/해제/복사 방법을 살펴볼 수 있다.

<br>

Linear memory는 `cudaMallocPitch()`와 `cudaMalloc3D()`를 통해서도 할당될 수 있다. 이 함수들은 2D array 또는 3D array를 할당할 때 권장되는데, **alignment requirement**를 만족시키고자 패딩(padding) 등을 추가할 때 적절하게 사용될 수 있다. 이를 잘 활용하면 최상의 성능을 얻을 수 있다고 한다. 아래 코드는 `cudaMallocPitch()`를 사용하여 `width` x `height` 2D 배열을 할당하고, device code에서 어떻게 배열 요소를 순회할 수 있는지를 보여준다.
```c++
// host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;

cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// device code
__global__
void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; r++) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; c++) {
            float element = row[c];
        }
    }
}
```

> `cudaMallocPitch()`에 대한 자세한 설명은 [link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)를 참조바람
>
> 이 함수는 주어진 행이 coalescing access를 위한 alignment requirements를 충족시키기 위해 padding을 추가할 수 있고, 이 함수에서 반환되는 `pitch`는 할당된 바이트의 폭(width)이다.

<br>

아래 코드는 `width` x `height` x `depth`의 3D array를 할당하고 device code에서 배열 요소를 순회하는 방법을 보여준다.

```c++
// host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// device code
__global__
void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;

    for (int z = 0; z < depth; z++) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; y++) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; x++) {
                float element = row[x];
            }
        }
    }
}
```

<br>

`cudaGetSymbolAddress()`는 global memory 공간에 선언된 변수가 할당된 메모리를 가리키는 포인터 주소를 알고싶을 때 사용한다. 할당된 메모리의 크기는 `cudaGetSymbolSize()`를 통해 얻을 수 있다.

<br>


# References

- [NVIDIA CUDA Documentations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)