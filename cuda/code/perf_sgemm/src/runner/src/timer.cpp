#include <timer.hpp>
#include <algorithm>
#include <numeric>

Timer::Timer()
{
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
}

Timer::~Timer()
{
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
}

void Timer::tic(cudaStream_t stream)
{
    CUDA_ERROR_CHECK(cudaEventRecord(start, stream));
}

float Timer::toc(cudaStream_t stream)
{
    CUDA_ERROR_CHECK(cudaEventRecord(stop, stream));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

    float msec = 0.f;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));
    record.push_back(msec);

    return msec;
}

void Timer::reset()
{
    record.clear();
}

float Timer::max()
{
    if (record.empty()) return std::numeric_limits<float>::infinity();
    return *std::max_element(record.cbegin(), record.cend());
}

float Timer::min()
{
    if (record.empty()) return std::numeric_limits<float>::infinity();
    return *std::min_element(record.cbegin(), record.cend());
}

float Timer::med()
{
    if (record.empty()) return std::numeric_limits<float>::infinity();

    std::sort(record.begin(), record.end());
    int const m = record.size() / 2;
    if (record.size() % 2) {
        return record[m];
    }
    return (record[m - 1] + record[m]) / 2;
}

float Timer::avg()
{
    if (record.empty()) return std::numeric_limits<float>::infinity();
    float total = std::accumulate(record.cbegin(), record.cend(), 0.f, std::plus<>());
    return total / record.size();
}