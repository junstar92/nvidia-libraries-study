# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
- [Using the Runtime API to Query GPU Information](#using-the-runtime-api-to-query-gpu-information)
  - [Determining the Best GPU](#determining-the-best-gpu)
- [Using nvidia-smi to Query GPU Information](#using-nvidia-smi-to-query-gpu-information)
- [Setting Devices at Runtime](#setting-devices-at-runtime)
- [References](#references)

<br>

# Intro

NVIDIA는 GPU device를 쿼리하고 관리할 수 있는 몇 가지 수단을 제공한다. 어떻게 device 정보를 쿼리하는지 아는 것은 런타임에 커널의 execution configuration을 설정하는데 도움이 된다.

여기서는 아래 두 개의 섹션을 통해서 GPU device 정보에 대해 쿼리할 수 있는 방법에 대해 알아본다.

- CUDA Runtime API functions
- NVIDIA System Management Interface(`nvidia-smi`) command-line utility

<br>

# Using the Runtime API to Query GPU Information

아래의 CUDA runtime API를 사용하여 시스템에서 사용되는 모든 GPU 장치들에 대한 정보를 쿼리할 수 있다.

```c++
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
```

GPU 장치의 속성은 [`cudaDeviceProp`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp) 구조체에 리턴된다.

NVIDIA에서 제공하는 CUDA 샘플 코드에는 [deviceQuery](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery)를 제공한다. 이 코드를 컴파일하여 실행하면 아래와 같은 결과를 얻을 수 있다.
```
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3080"
  CUDA Driver Version / Runtime Version          12.0 / 11.8
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 9987 MBytes (10472390656 bytes)
  (068) Multiprocessors, (128) CUDA Cores/MP:    8704 CUDA Cores
  GPU Max Clock rate:                            1710 MHz (1.71 GHz)
  Memory Clock rate:                             9501 Mhz
  Memory Bus Width:                              320-bit
  L2 Cache Size:                                 5242880 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 9 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.0, CUDA Runtime Version = 11.8, NumDevs = 1
Result = PASS
```

꽤 많은 정보들을 담고 있는데, 필요한 것들만 추려서 따로 출력해주도록 [device_query.cpp](/cuda/code/device_query/device_query.cpp)를 올려두었으니 참조해도 좋을 것 같다.

## Determining the Best GPU

만약 시스템에 서로 다른 종류의 GPU가 여러 개 달려있는 경우가 있을 수 있다. 이런 경우, 커널을 실행할 최상의 GPU를 선택하는 것이 중요하다. 최상의 GPU를 선택하는 한 가지 방법은 GPU의 multiprocessors(MPs) 갯수를 식별하는 것이다. 예를 들면, 아래와 같이 코드를 작성하여 가장 연산 능력이 뛰어난 장치를 선택할 수 있다.

```c++
int num_devices = 0;
cudaGetDeviceCount(&num_devices);

if (num_devices > 1) {
    int max_mps = 0, max_dev = 0;
    for (int dev = 0; dev < num_devices; dev++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, dev);
        if (max_mps < props.multiProcessorCount) {
            max_mps = props.multiProcessorCount;
            max_dev = dev;
        }
    }
    cudaSetDevice(max_dev);
}
```

<br>

# Using nvidia-smi to Query GPU Information

Command-line tool인 `nvidia-smi`를 사용하면 GPU 장치를 관리하고 모니터링할 수 있다. 예를 들어, 시스템에 몇 개의 GPU가 설치되어 있는지 확인하려면 아래와 같이 입력하면 된다.
```
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-382f23c1-5160-01e2-3291-ff9628930b70)
```

현재 필자의 시스템에는 하나의 GPU (GPU 0)만 달려있기 때문에 하나만 출력하고 있다.

만약 첫 번째 GPU(GPU 0)에 대해 자세한 정보를 보려면, 아래와 같이 커맨드를 입력하면 된다.
```
$ nvidia-smi -q -i 0
```

이때 출력되는 정보가 많은데, 몇 가지 정보만 선택해서 표시할 수도 있다. 예를 들어, GPU 장치의 메모리 정보만 보고 싶다면, 다음과 같이 커맨드를 입력해주면 된다.
```
$ nvidia-smi -q -i 0 -d MEMORY

GPU 00000000:09:00.0
    FB Memory Usage
        Total                             : 10240 MiB
        Reserved                          : 252 MiB
        Used                              : 352 MiB
        Free                              : 9635 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 27 MiB
        Free                              : 229 MiB
```

`-d(--display=)` 인자로 추가할 수 있는 명령으로는 

- MEMORY
- UTILIZATION
- ECC
- TEMPERATURE
- POWER
- CLOCK
- COMPUTE
- PIDS
- PERFORMANCE
- SUPPORTED_CLOCKS
- PAGE_RETIERMENT
- ACCOUNTING
- ENCODER_STATS
- SUPPORTED_GPU_TARGET_TEMP
- VOLTAGE
- FBC_STATS
- POW_REMAPPER

가 있다.

<br>

# Setting Devices at Runtime

만약 시스템에서 여러 개의 GPU가 지원된다면, 이 GPU들의 device ID는 0번부터 N-1번까지 부여된다. 이때, `CUDA_VISIBLE_DEVICES` 환경 변수를 사용하면, 프로그램 변경없이 런타임에서 어떤 GPU를 사용할 지 지정할 수 있다.

만약 `CUDA_VISIBLE_DEVICES=2`로 지정한다면, 프로그램 내에서 다른 GPU는 off시키고, device 2를 device 0으로 표시하게 된다.

또한, `CUDA_VISIBLE_DEVICES=2,3`과 같이 여러 GPU를 지정할 수도 있다. 이렇게 지정하면, device 2, 3은 런타임 프로그램 내에서 device 0, 1로 각각 매핑된다.


<br>

# References

- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher