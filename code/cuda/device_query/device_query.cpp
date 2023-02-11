/*****************************************************************************
 * File:        device_query.cpp
 * Description: Qeury device(s) in current system
 *              
 * Compile:     nvcc -o device_query device_query.cpp
 * Run:         ./device_query
 *****************************************************************************/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

int main(int argc, char** argv)
{
    printf("> CUDA Device Query (Runtime API) version (CUDAAR static linking)\n\n");

    int dev_count = 0;
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&dev_count));

    if (dev_count == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", dev_count);
    }

    int driver_ver = 0, runtime_ver = 0;

    for (int dev = 0; dev < dev_count; dev++) {
        CUDA_ERROR_CHECK(cudaSetDevice(dev));
        cudaDeviceProp dev_prop;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&dev_prop, dev));

        printf("\n> Device %d: \"%s\"\n", dev, dev_prop.name);

        CUDA_ERROR_CHECK(cudaDriverGetVersion(&driver_ver));
        CUDA_ERROR_CHECK(cudaRuntimeGetVersion(&runtime_ver));
        printf("  CUDA Driver Version / Runtime Version             %d.%d / %d.%d\n",
            driver_ver / 1000, (driver_ver % 100) / 10,
            runtime_ver / 1000, (runtime_ver % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:       %d.%d\n",
            dev_prop.major, dev_prop.minor);

        printf("  Total amount of global memory:                    %.0f MBytes (%llu Bytes)\n",
            static_cast<float>(dev_prop.totalGlobalMem / 1048576.f),
            static_cast<unsigned long long>(dev_prop.totalGlobalMem));
        
        printf("  GPU Clock rate:                                   %.0f MHz (%0.2f GHz)\n",
            dev_prop.clockRate * 1e-3f, dev_prop.clockRate * 1e-6f);
        printf("  Memory Clock rate:                                %.0f MHz\n",
            dev_prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                                 %d-bit\n",
            dev_prop.memoryBusWidth);
        
        if (dev_prop.l2CacheSize) {
            printf("  L2 Cache Size:                                    %.0f MBytes (%d Bytes)\n",
                static_cast<float>(dev_prop.l2CacheSize / 1048576.f), dev_prop.l2CacheSize);
        }

        printf("  Total amount of constant memory:                  %lu Bytes\n", dev_prop.totalConstMem);
        printf("  Total amount of shared memory per block:          %lu Bytes\n", dev_prop.sharedMemPerBlock);
        printf("  Total number of registers available per block:    %d\n", dev_prop.regsPerBlock);
        printf("  Warp size:                                        %d\n", dev_prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:     %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:              %d\n", dev_prop.maxThreadsPerBlock);
        printf("  Maximum number of warps per multiprocessors:      %d\n", dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize);
        printf("  Max dimension size of a thread block (x,y,z):     (%d, %d, %d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size (x,y,z):        (%d, %d, %d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                             %lu Bytes\n", dev_prop.memPitch);
        printf("  Maximum texture dimension size (x,y,z):           1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            dev_prop.maxTexture1D,
            dev_prop.maxTexture2D[0], dev_prop.maxTexture2D[1],
            dev_prop.maxTexture3D[0], dev_prop.maxTexture3D[1], dev_prop.maxTexture3D[2]);
        printf("  Texture alignment:                                %ld\n", dev_prop.textureAlignment);
        printf("  Concurrent copy and kernel execution:             %s with %d copy engine(s)\n",
            (dev_prop.deviceOverlap ? "Yes" : "No"), dev_prop.asyncEngineCount);
        printf("  Integrated GPU sharing Host Memory:               %s\n", dev_prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:          %s\n", dev_prop.canMapHostMemory ? "Yes" : "No");
        printf("  Device supports Unified Addressing (UVA):         %s\n", dev_prop.unifiedAddressing ? "Yes" : "No");
    }

    return 0;
}