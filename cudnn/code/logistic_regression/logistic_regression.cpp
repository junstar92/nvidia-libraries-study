#include <iostream>
#include <string>
#include <fstream>

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

const std::string FC_WEIGHT_BIN_NAME = "weight.bin";
const std::string FC_BIAS_BIN_NAME = "bias.bin";
const std::string MNIST_0_BIN_NAME = "0.bin";
const std::string MNIST_1_BIN_NAME = "1.bin";

void load_binary(float* ptr, const int count, const char* filename)
{
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(ptr), sizeof(ptr) * count);
}

void show_digit(float* img, const int h, const int w)
{
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%c", img[i * w + j] >= 0.5f ? '*' : '.');
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    int batch_size = 1;
    int in_dim = 28 * 28;
    int out_dim = 1;

    // load input
    float* input1 = (float*)malloc(sizeof(float) * in_dim);
    float* input2 = (float*)malloc(sizeof(float) * in_dim);
    load_binary(input1, in_dim, MNIST_0_BIN_NAME.c_str());
    load_binary(input2, in_dim, MNIST_1_BIN_NAME.c_str());

    // load weight and bias from binary files
    float* weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    float* bias = (float*)malloc(sizeof(float) * out_dim);
    load_binary(weight, in_dim * out_dim, FC_WEIGHT_BIN_NAME.c_str());
    load_binary(bias, out_dim, FC_BIAS_BIN_NAME.c_str());

    /*** cuDNN handle, descriptors setting ***/

    // create cudnn handle
    cudnnHandle_t handle;
    CUDNN_ERROR_CHECK(cudnnCreate(&handle));

    // create/set input, output descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    int input_shape[] = { batch_size, in_dim, 1, 1 };
    int output_shape[] = { batch_size, out_dim, 1, 1 };
    int input_stride[] = { in_dim, 1, 1, 1};
    int output_stride[] = { out_dim, 1, 1, 1};
    CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_ERROR_CHECK(cudnnSetTensorNdDescriptor(input_desc, data_type, 4, input_shape, input_stride));
    CUDNN_ERROR_CHECK(cudnnSetTensorNdDescriptor(output_desc, data_type, 4, output_shape, output_stride));

    // create/set weight, bias descriptors
    cudnnFilterDescriptor_t weight_desc;
    cudnnTensorDescriptor_t bias_desc;
    int weight_shape[] = { out_dim, in_dim, 1, 1};
    int bias_shape[] = { out_dim, 1, 1, 1 };
    int bias_stride[] = { out_dim, 1, 1, 1 };
    CUDNN_ERROR_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
    CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_ERROR_CHECK(cudnnSetFilterNdDescriptor(weight_desc, data_type, CUDNN_TENSOR_NCHW, 4, weight_shape));
    CUDNN_ERROR_CHECK(cudnnSetTensorNdDescriptor(bias_desc, data_type, 4, bias_shape, bias_stride));

    // create/set conv descriptor for fc layer (weight matmul)
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_ERROR_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    const int conv_ndims = 2;
    int padding[conv_ndims] = {0,0};
    int stride[conv_ndims] = {1,1};
    int dilation[conv_ndims] = {1,1};
    CUDNN_ERROR_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc, conv_ndims, padding, stride, dilation, CUDNN_CROSS_CORRELATION, data_type));

    // check output dimension after convolution op
    int output_shape_by_conv[4] = {};
    CUDNN_ERROR_CHECK(cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_desc, weight_desc, 4, output_shape_by_conv));
    printf("Input shape            : (%d, %d, %d, %d)\n", 1, in_dim, 1, 1);
    printf("Output shape after conv: (%d, %d, %d, %d)\n", output_shape_by_conv[0], output_shape_by_conv[1], output_shape_by_conv[2], output_shape_by_conv[3]);
    
    int requested_algo_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returned_algo_count = -1;
    cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CUDNN_ERROR_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, weight_desc, conv_desc, output_desc, requested_algo_count, &returned_algo_count, results));
    printf("\nTesting cudnnGetConvolutionForwardAlgorithm_v7...\n");
    for (int i = 0; i < returned_algo_count; i++) {
        printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
            cudnnGetErrorString(results[i].status), 
            results[i].algo, results[i].time, 
            (unsigned long long)results[i].memory);
    }
    printf("\n");

    CUDNN_ERROR_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle,
        input_desc, weight_desc, conv_desc, output_desc,
        requested_algo_count, &returned_algo_count,
        results
    ));
    printf("\nTesting cudnnFindConvolutionForwardAlgorithm...\n");
    for(int i = 0; i < returned_algo_count; ++i){
        printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
            cudnnGetErrorString(results[i].status), 
            results[i].algo, results[i].time, 
            (unsigned long long)results[i].memory);
    }
    printf("\n");

    // set algorithm and memory
    auto algo = results[0].algo;
    size_t workspace_size = results[0].memory;
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_ERROR_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }
    
    // create/set activation descriptor for sigmoid function
    cudnnActivationDescriptor_t sigmoid_desc;
    CUDNN_ERROR_CHECK(cudnnCreateActivationDescriptor(&sigmoid_desc));
    CUDNN_ERROR_CHECK(cudnnSetActivationDescriptor(sigmoid_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));


    /*** inference ***/

    // allocate device memory for input/output
    void *d_input, *d_output, *d_weight, *d_bias;
    CUDA_ERROR_CHECK(cudaMalloc(&d_input, sizeof(float) * in_dim));
    CUDA_ERROR_CHECK(cudaMalloc(&d_output, sizeof(float) * out_dim));
    CUDA_ERROR_CHECK(cudaMalloc(&d_weight, sizeof(float) * out_dim * in_dim));
    CUDA_ERROR_CHECK(cudaMalloc(&d_bias, sizeof(float) * out_dim));

    // memcpy from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_input, input1, sizeof(float) * in_dim, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_weight, weight, sizeof(float) * out_dim * in_dim, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_bias, bias, sizeof(float) * out_dim, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemset(d_output, 0, sizeof(float) * out_dim));

    // inference for digit 0
    const float alpha1 = 1.f;
    const float alpha2 = 0.f;
    
    CUDNN_ERROR_CHECK(cudnnConvolutionForward(
        handle,
        static_cast<const void*>(&alpha1),
        input_desc, d_input,
        weight_desc, d_weight,
        conv_desc, algo, d_workspace, workspace_size,
        static_cast<const void*>(&alpha2),
        output_desc, d_output
    ));
    CUDNN_ERROR_CHECK(cudnnAddTensor(
        handle,
        static_cast<const void*>(&alpha1),
        bias_desc, d_bias,
        static_cast<const void*>(&alpha1),
        output_desc, d_output
    ));
    CUDNN_ERROR_CHECK(cudnnActivationForward(
        handle,
        sigmoid_desc,
        static_cast<const void*>(&alpha1),
        output_desc, d_output,
        static_cast<const void*>(&alpha2),
        output_desc, d_output
    ));

    // validate output
    float output;
    CUDA_ERROR_CHECK(cudaMemcpy(&output, d_output, sizeof(float) * out_dim, cudaMemcpyDeviceToHost));
    show_digit(input1, 28, 28);
    printf("Output: %.3f -> Digit %d\n\n", output, output >= 0.5f ? 1 : 0);

    // inference for digit 1
    CUDA_ERROR_CHECK(cudaMemcpy(d_input, input2, sizeof(float) * in_dim, cudaMemcpyHostToDevice));

    CUDNN_ERROR_CHECK(cudnnConvolutionBiasActivationForward(
        handle,
        static_cast<const void*>(&alpha1),
        input_desc,
        d_input,
        weight_desc,
        d_weight,
        conv_desc,
        algo,
        d_workspace,
        workspace_size,
        static_cast<const void*>(&alpha2),
        output_desc,
        d_output,
        bias_desc,
        d_bias,
        sigmoid_desc,
        output_desc,
        d_output
    ));

    // validate output
    CUDA_ERROR_CHECK(cudaMemcpy(&output, d_output, sizeof(float) * out_dim, cudaMemcpyDeviceToHost));
    show_digit(input2, 28, 28);
    printf("Output: %.3f -> Digit %d\n", output, output >= 0.5f ? 1 : 0);


    /*** free resources ***/
    CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_ERROR_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
    CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_ERROR_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_ERROR_CHECK(cudnnDestroyActivationDescriptor(sigmoid_desc));
    CUDNN_ERROR_CHECK(cudnnDestroy(handle));

    free(input1);
    free(input2);
    free(weight);
    free(bias);
    if (d_workspace != nullptr) CUDA_ERROR_CHECK(cudaFree(d_workspace));
    CUDA_ERROR_CHECK(cudaFree(d_input));
    CUDA_ERROR_CHECK(cudaFree(d_output));
    CUDA_ERROR_CHECK(cudaFree(d_weight));
    CUDA_ERROR_CHECK(cudaFree(d_bias));

    return 0;
}