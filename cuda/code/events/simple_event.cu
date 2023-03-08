/*****************************************************************************
 * File:        simple_event.cu
 * Description: This example demonstrates inter-stream dependencies using 
 *              cudaStreamWaitEvent. It launches 4 kernels in each of streams.
 *              An event is recorded at the completion of each stream.
 *              cudaStreamWaitEvent is then called on that event and the last
 *              stream to force all computation in the final stream to only
 *              execute when all other streams have completed.
 *              
 * Compile:     nvcc -O3 -o simple_event simple_event.cu
 * Run:         ./simple_event <n>
 *                  <n> : the number of streams (default: 4)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define N 1

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
    dim3 block(128);
    dim3 grid(32);

    // allocate CUDA events
    cudaEvent_t* kernel_events = (cudaEvent_t*)malloc(num_streams * sizeof(cudaEvent_t));
    for (int i = 0; i < num_streams; i++) {
        CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&kernel_events[i], cudaEventDisableTiming));
    }

    for (int i = 0; i < num_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();

        CUDA_ERROR_CHECK(cudaEventRecord(kernel_events[i], streams[i]));
        CUDA_ERROR_CHECK(cudaStreamWaitEvent(streams[num_streams - 1], kernel_events[i]));
    }

    // release all streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    // release all events
    for (int i = 0; i < num_streams; i++) {
        CUDA_ERROR_CHECK(cudaEventDestroy(kernel_events[i]));
    }
    free(kernel_events);

    CUDA_ERROR_CHECK(cudaDeviceReset());
    return 0;
}