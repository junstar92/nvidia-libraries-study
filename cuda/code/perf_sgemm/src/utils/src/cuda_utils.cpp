#include <cuda_utils.hpp>
#include <utils.hpp>

inline int convert_sm2cores(int const major, int const minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
        return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    return nGpuArchCoresPerSM[index - 1].Cores;
}

void cuda_device_info(int const device_id)
{
    cudaDeviceProp prop{};
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("Device %d: \"%s\"\n", device_id, prop.name);

    int driver_ver, runtime_ver;
    CUDA_ERROR_CHECK(cudaDriverGetVersion(&driver_ver));
    CUDA_ERROR_CHECK(cudaRuntimeGetVersion(&runtime_ver));
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
            driver_ver / 1000, (driver_ver % 100) / 10,
            runtime_ver / 1000, (runtime_ver % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
            prop.major, prop.minor);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
            prop.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
            prop.maxThreadsPerBlock);
    printf("  Warp size:                                     %d\n",
           prop.warpSize);
    printf("  Maximum number of warps per multiprocessor:    %d\n",
            prop.maxThreadsPerMultiProcessor/prop.warpSize);
    printf("  (%3d) Multiprocessors, (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
            prop.multiProcessorCount, convert_sm2cores(prop.major, prop.minor), convert_sm2cores(prop.major, prop.minor) * prop.multiProcessorCount);
    printf("  Total amount of global memory:                 %.0f MBytes (%llu Bytes)\n",
            static_cast<float>(prop.totalGlobalMem / 1048576.0f), static_cast<unsigned long long>(prop.totalGlobalMem));
    printf("  Total amount of constant memory:               %zu Bytes (%zu KBytes)\n",
            prop.totalConstMem, prop.totalConstMem / 1024);
    printf("  Total amount of shared memory per block:       %zu Bytes (%zu KBytes)\n",
            prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024);
    printf("  Total shared memory per multiprocessor:        %zu Bytes (%zu KBytes)\n",
            prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024);
    printf("  Total number of registers available per block: %d\n",
            prop.regsPerBlock);
    printf("  Total number of registers available per SM:    %d\n",
            prop.regsPerMultiprocessor);

    printf("\n");
}