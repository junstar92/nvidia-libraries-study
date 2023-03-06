/*****************************************************************************
 * File:        stream_callbacks.cu
 * Description: This example demonstrates the stream callbacks. Callback function
 *              in a stream will be called after all preceding tasks in the stream
 *              complete
 * .
 *              kernel_1, kernel_2, kernel_3, and kernel_4 simply implement
 *              identical dummy computation.
 *              
 * Compile:     nvcc -O3 -o stream_callbacks stream_callbacks.cu
 * Run:         ./stream_callbacks <n>
 *                  <n> : the number of streams (default: 4)
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(err); \
    }

#define N 100000

// void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* data) // for cudaStreamAddCallback
void CUDART_CB myCallback(void* data)
{
    printf("Callback from stream %d\n", *((int*)data));
}

__global__
void kernel_1(int stream_id)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
    printf("[stream %d] kernel_1\n", stream_id);
}

__global__
void kernel_2(int stream_id)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
    printf("[stream %d] kernel_2\n", stream_id);
}

__global__
void kernel_3(int stream_id)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
    printf("[stream %d] kernel_3\n", stream_id);
}

__global__
void kernel_4(int stream_id)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.1);
    }
    printf("[stream %d] kernel_4\n", stream_id);
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

    int* stream_ids = (int*)malloc(num_streams * sizeof(int));
    for (int i = 0; i < num_streams; i++) {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>(i);
        kernel_2<<<grid, block, 0, streams[i]>>>(i);
        kernel_3<<<grid, block, 0, streams[i]>>>(i);
        kernel_4<<<grid, block, 0, streams[i]>>>(i);
        //CUDA_ERROR_CHECK(cudaStreamAddCallback(streams[i], myCallback, &stream_ids[i], 0));
        CUDA_ERROR_CHECK(cudaLaunchHostFunc(streams[i], myCallback, &stream_ids[i]));
    }

    CUDA_ERROR_CHECK(cudaEventRecord(end, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize(end));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, end));
    printf("Measured time for parallel execution: %.3f ms\n", msec);

    free(stream_ids);

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