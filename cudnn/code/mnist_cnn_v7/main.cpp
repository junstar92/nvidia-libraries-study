// compile ccommand: nvcc -o mnist_cnn -lcudnn -I. main.cpp

#include <mnist_cnn.h>
#include <algorithm>
#include <math.h>

void show_digit(float* img, const int h, const int w)
{
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%c", img[i * w + j] >= 0.5f ? '*' : '.');
        }
        printf("\n");
    }
}

float get_prob(const float* probs, int idx)
{
    float sum = 0.f;
    for (int i = 0; i < 10; i++) {
        sum += exp(probs[i]);
    }

    return exp(probs[idx]) / sum;
}

int main(int argc, char** argv)
{
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    // create network and intialization
    MnistCNN network(data_type);
    Dims4 input_dims{1,1,28,28};
    network.init(input_dims);

    // malloc memory for inference
    float* digit = (float*)malloc(sizeof(float) * 28 * 28);
    float output[10];
    void *d_input, *d_output;
    CUDA_ERROR_CHECK(cudaMalloc(&d_input, sizeof(float) * 28 * 28));
    CUDA_ERROR_CHECK(cudaMalloc(&d_output, sizeof(float) * 10));

    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    float msec = 0.f;

    for (int i = 0; i < 10; i++) {
        // get input data
        std::string filename = "digits/" + std::to_string(i) + ".bin";
        loadBinary((void*)digit, 28 * 28, filename.c_str());
        show_digit(digit, 28, 28);
        cudaMemcpy(d_input, digit, sizeof(float) * 28 * 28, cudaMemcpyHostToDevice);

        // inference
        CUDA_ERROR_CHECK(cudaEventRecord(start));
        network.forward(d_input, d_output);
        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&msec, start, stop));

        // extract output
        cudaMemcpy(output, d_output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        
        auto iter = std::max_element(output, output + 10);
        int output_digit = std::distance(output, iter);
        std::cout << "Digit: " << output_digit << " (" << get_prob(output, output_digit) << ")\n";
        std::cout << "Elapsed Time: " << msec << " ms\n\n";
    }

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    CUDA_ERROR_CHECK(cudaFree(d_input));
    CUDA_ERROR_CHECK(cudaFree(d_output));
    free(digit);

    return 0;
}