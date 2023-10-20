#pragma once
#include <string>
#include <vector>
#include <functional>
#include <limits>

class Runner
{
public:
    struct Result
    {
        int size{};
        float max, min, med, avg;

        Result(int size) : size{size} {
            max = std::numeric_limits<float>::infinity();
            min = std::numeric_limits<float>::infinity();
            med = std::numeric_limits<float>::infinity();
            avg = std::numeric_limits<float>::infinity();
        }
    };
    struct Kernel
    {
        std::string kernel_name;
        std::vector<Result> results;

        Kernel(std::string_view kernel_name, std::vector<int> const& sizes) : kernel_name{kernel_name} {
            for (int size : sizes) {
                results.emplace_back(size);
            }
        }
    };

public:
    Runner(std::vector<int> const& sizes, int num_warmup = 25, int num_iteration = 100);
    virtual ~Runner() {}

    void init();
    void run();
    void print_results() const;

private:
    void run_kernels(int size);
    
    int num_warmup{25};
    int num_iteration{100};
    int num_kernels{};

    std::vector<int> sizes;
    std::vector<Kernel> kernels;
};