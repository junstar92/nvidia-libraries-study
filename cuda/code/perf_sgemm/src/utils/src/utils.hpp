#pragma once
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) cuda_error_check((err), #err, __FILE__, __LINE__)

inline void cuda_error_check(cudaError_t err, char const* const func, char const* const file, int const num_line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error(" <<  func << "): " << cudaGetErrorString(err) << " at " << file << ":" << num_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}