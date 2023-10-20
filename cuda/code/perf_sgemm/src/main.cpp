#include <iostream>
#include <runner.hpp>

int main(int argc, char** argv)
{
    std::vector<int> sizes;
    int num_warmup = 25;
    int num_iteration = 100;
    if (argc > 1) {
        sizes.push_back(std::stoi(argv[1]));
    }
    else {
        for (int i = 2; i < 33; i++) {
            sizes.push_back(128 * i);
        }
    }
    if (argc > 2) {
        num_warmup = std::stoi(argv[2]);
    }
    if (argc > 3) {
        num_iteration = std::stoi(argv[3]);
    }

    Runner runner(sizes, num_warmup, num_iteration);
    runner.init();
    runner.run();
    runner.print_results();

    return EXIT_SUCCESS;
}