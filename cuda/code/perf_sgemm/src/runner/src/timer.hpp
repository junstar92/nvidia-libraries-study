#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <utils.hpp>

class Timer
{
public:
    Timer();
    virtual ~Timer();

    void tic(cudaStream_t stream = nullptr);
    float toc(cudaStream_t stream = nullptr);
    void reset();

    float max();
    float min();
    float med();
    float avg();


private:
    cudaEvent_t start{}, stop{};
    std::vector<float> record{};
};