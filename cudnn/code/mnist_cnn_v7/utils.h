#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_ERROR_CHECK(status) \
    if (status != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(status), __LINE__, __FILE__); \
        cudaDeviceReset(); \
        exit(status); \
    }

#define CUDNN_ERROR_CHECK(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudnnGetErrorString(status), __LINE__, __FILE__); \
        cudaDeviceReset(); \
        exit(status); \
    } \
}

struct Dims4 {
    int n;
    int c;
    int h;
    int w;
};