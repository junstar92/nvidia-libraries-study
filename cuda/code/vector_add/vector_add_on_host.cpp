/*****************************************************************************
 * File:        vector_add_on_host.cpp
 * Description: Sequential Vector Addition. A + B = C
 *              
 * Compile:     nvcc -o vector_add_on_host vector_add_on_host.cpp
 * Run:         ./vector_add_on_host <n>
 *                  <n> : the number of vector elements = n power(s) of 2
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

void addVectorOnHost(float const* a, float const* b, float* c, const int num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i];
    }
}

void initVector(float* vec, const int num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        vec[i] = rand() / (float)RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    int pow = strtol(argv[1], NULL, 10);
    int num_elements = 1 << pow;

    printf("[Vector addition of %d elements on Host]\n", num_elements);

    float *a, *b, *c;
    a = static_cast<float*>(malloc(sizeof(float) * num_elements));
    b = static_cast<float*>(malloc(sizeof(float) * num_elements));
    c = static_cast<float*>(malloc(sizeof(float) * num_elements));

    initVector(a, num_elements);
    initVector(b, num_elements);

    auto start = chrono::high_resolution_clock::now();
    addVectorOnHost(a, b, c, num_elements);
    auto end = chrono::high_resolution_clock::now();

    float msec = static_cast<float>(chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.f;
    printf("Time: %.3f msec\n", msec);

    free(a);
    free(b);
    free(c);

    return 0;
}