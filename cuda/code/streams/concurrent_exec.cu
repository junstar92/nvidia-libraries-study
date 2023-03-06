/*****************************************************************************
 * File:        concurrent_exec.cu
 * Description: This example demonstrates concurrent kernel execution by submitting
 *              tasks to a CUDA stream in depth-first order.
 *              kernel_1, kernel_2, kernel_3, and kernel_4 simply implement
 *              identical dummy computation.
 * 
 *              Concurrent kernel execution can be visualized in nsight systems.
 *              
 * Compile:     nvcc -O3 -o concurrent_exec concurrent_exec.cu
 * Run:         ./concurrent_exec <n>
 *                  <n> : the number of streams (default: 4)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define N 1000

__global__
void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
        printf("%f\n", sum);
    }
}

__global__
void kernel_2()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
        printf("%f\n", sum);
    }
}

__global__
void kernel_3()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
        printf("%f\n", sum);
    }
}

__global__
void kernel_4()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
        printf("%f\n", sum);
    }
}

int main(int argc, char** argv)
{
    // get the number of streams
    int num_streams = 4;
    if (argc > 1) num_streams = atoi(argv[1]);

    // set up max connectioin (hyper-q)
    // char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    // setenv (iname, "1", 1);
    // char *ivalue =  getenv (iname);
    // printf ("%s = %s\n", iname, ivalue);

    // setup device
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("> At device %d: %s with num_streams=%d\n", dev, prop.name, num_streams);
    CUDA_ERROR_CHECK(cudaSetDevice(dev));

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", prop.major, prop.minor, prop.multiProcessorCount);

    // allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        CUDA_ERROR_CHECK(cudaStreamCreate(&streams[i]));
    }

    // setup execution configuration
    dim3 block(1);
    dim3 grid(1);

    // CUDA events
    cudaEvent_t start, end;
    float msec;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));

    // launch kernels with streams
    CUDA_ERROR_CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < num_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    CUDA_ERROR_CHECK(cudaEventRecord(end, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("Measured time for parallel execution: %.3f ms\n", msec);

    // release all streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));
    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}