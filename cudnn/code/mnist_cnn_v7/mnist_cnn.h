#pragma once
#include <layer.h>
#include <vector>
#include <string>
#include <fstream>

static const std::string CONV1_WEIGHT = "params/conv1_weight.bin";
static const std::string CONV1_BIAS = "params/conv1_bias.bin";
static const std::string CONV2_WEIGHT = "params/conv2_weight.bin";
static const std::string CONV2_BIAS = "params/conv2_bias.bin";
static const std::string FC_WEIGHT = "params/fc_weight.bin";
static const std::string FC_BIAS = "params/fc_bias.bin";

void loadBinary(void* ptr, const int count, const char* filename)
{
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(ptr), count * 4); // support only fp32
}

class MnistCNN
{
public:
    MnistCNN(cudnnDataType_t type) : m_data_type(type) {
        CUDNN_ERROR_CHECK(cudnnCreate(&m_handle));

        // build MnistCNN Network
        auto conv1 = addConv2d(1, 32, 3, 1, 1, 1, 1, true, CUDNN_ACTIVATION_RELU); // conv + bias add + relu
        auto maxpool1 = addPooling2d(2, 2); // max pooling
        auto conv2 = addConv2d(32, 64, 3, 1, 1, 1, 1, true, CUDNN_ACTIVATION_RELU); // conv + bias add + relu
        auto maxpool2 = addPooling2d(2, 2); // max pooling
        auto fc = addConv2d(7 * 7 * 64, 10, 1, 1); // fully connected

        // load weight and bias
        m_conv1_weight = (void*)malloc(32 * 3 * 3 * sizeof(float));
        m_conv1_bias = (void*)malloc(32 * sizeof(float));
        m_conv2_weight = (void*)malloc(64 * 32 * 3 * 3 * sizeof(float));
        m_conv2_bias = (void*)malloc(64 * sizeof(float));
        m_fc_weight = (void*)malloc(7 * 7 * 64 * 10 * sizeof(float));
        m_fc_bias = (void*)malloc(10 * sizeof(float));
        CUDA_ERROR_CHECK(cudaMalloc(&m_conv1_weight_dev, 32 * 3 * 3 * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_conv1_bias_dev, 32 * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_conv2_weight_dev, 64 * 32 * 3 * 3 * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_conv2_bias_dev, 64 * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_fc_weight_dev, 7 * 7 * 64 * 10 * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_fc_bias_dev, 10 * sizeof(float)));

        loadBinary(m_conv1_weight, 32 * 3 * 3, CONV1_WEIGHT.c_str());
        loadBinary(m_conv1_bias, 32, CONV1_BIAS.c_str());
        loadBinary(m_conv2_weight, 64 * 32 * 3 * 3, CONV2_WEIGHT.c_str());
        loadBinary(m_conv2_bias, 64, CONV2_BIAS.c_str());
        loadBinary(m_fc_weight, 7 * 7 * 64 * 10, FC_WEIGHT.c_str());
        loadBinary(m_fc_bias, 10, FC_BIAS.c_str());

        CUDA_ERROR_CHECK(cudaMemcpy(m_conv1_weight_dev, m_conv1_weight, 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_conv1_bias_dev, m_conv1_bias, 32 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_conv2_weight_dev, m_conv2_weight, 64 * 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_conv2_bias_dev, m_conv2_bias, 64 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_fc_weight_dev, m_fc_weight, 7 * 7 * 64 * 10 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_fc_bias_dev, m_fc_bias, 10 * sizeof(float), cudaMemcpyHostToDevice));
        
        conv1->getParameters(m_conv1_weight_dev, m_conv1_bias_dev);
        conv2->getParameters(m_conv2_weight_dev, m_conv2_bias_dev);
        fc->getParameters(m_fc_weight_dev, m_fc_bias_dev);
    }
    virtual ~MnistCNN() {
        CUDNN_ERROR_CHECK(cudnnDestroy(m_handle));
        if (m_conv1_weight) free(m_conv1_weight);
        if (m_conv1_bias) free(m_conv1_bias);
        if (m_conv2_weight) free(m_conv2_weight);
        if (m_conv2_bias) free(m_conv2_bias);
        if (m_fc_weight) free(m_fc_weight);
        if (m_fc_bias) free(m_fc_bias);
        if (m_conv1_weight_dev) CUDA_ERROR_CHECK(cudaFree(m_conv1_weight_dev));
        if (m_conv1_bias_dev) CUDA_ERROR_CHECK(cudaFree(m_conv1_bias_dev));
        if (m_conv2_weight_dev) CUDA_ERROR_CHECK(cudaFree(m_conv2_weight_dev));
        if (m_conv2_bias_dev) CUDA_ERROR_CHECK(cudaFree(m_conv2_bias_dev));
        if (m_fc_weight_dev) CUDA_ERROR_CHECK(cudaFree(m_fc_weight_dev));
        if (m_fc_bias_dev) CUDA_ERROR_CHECK(cudaFree(m_fc_bias_dev));
        if (m_dev_mem[0]) CUDA_ERROR_CHECK(cudaFree(m_dev_mem[0]));
        if (m_dev_mem[1]) CUDA_ERROR_CHECK(cudaFree(m_dev_mem[1]));
    }

    void init(Dims4 input_dims) {
        size_t max_elem_count = input_dims.n * 64 * input_dims.h * input_dims.w;
        for (auto layer : m_layers) {
            layer->init(m_handle, input_dims, m_format, m_data_type);
            // std::cout << "In (" << input_dims.n << "," << input_dims.c << "," << input_dims.h << "," << input_dims.w << ") -> ";
            input_dims = layer->getOutputDimension();
            // std::cout << "Out (" << input_dims.n << "," << input_dims.c << "," << input_dims.h << "," << input_dims.w << ")\n";
        }
        size_t width = 4;
        CUDA_ERROR_CHECK(cudaMalloc(&m_dev_mem[0], max_elem_count * width));
        CUDA_ERROR_CHECK(cudaMalloc(&m_dev_mem[1], max_elem_count * width));
        m_init = true;
    }
    void forward(const void* src, void* dst) {
        int src_idx = 0, dst_idx = 1;
        int layer_idx = 0, layer_len = m_layers.size();

        m_layers[layer_idx]->forward(m_handle, src, m_dev_mem[src_idx]);
        for (layer_idx = 1; layer_idx < layer_len - 1; layer_idx++) {
            m_layers[layer_idx]->forward(m_handle, m_dev_mem[src_idx], m_dev_mem[dst_idx]);
            std::swap(src_idx, dst_idx);
        }
        m_layers[layer_len - 1]->forward(m_handle, m_dev_mem[src_idx], dst);
    }


private:
    bool m_init{false};
    std::vector<Layer*> m_layers{};

    void* m_dev_mem[2];
    void *m_conv1_weight, *m_conv1_bias, *m_conv2_weight, *m_conv2_bias, *m_fc_weight, *m_fc_bias;
    void *m_conv1_weight_dev, *m_conv1_bias_dev, *m_conv2_weight_dev, *m_conv2_bias_dev, *m_fc_weight_dev, *m_fc_bias_dev;

    cudnnHandle_t m_handle;
    cudnnTensorFormat_t m_format{CUDNN_TENSOR_NCHW};
    cudnnDataType_t m_data_type{CUDNN_DATA_FLOAT};

    Conv2d* addConv2d(int in_channel, int out_channel, int kernel_size, int stride = 1, int padding = 0, int dilation = 1, int groups = 1, bool bias = true, int activation = -1) {
        Conv2d* conv2d = new Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias, activation);
        m_layers.push_back(conv2d);

        return conv2d;
    }
    Pooling2d* addPooling2d(int kernel_size, int stride, int padding = 0) {
        Pooling2d* pooling2d = new Pooling2d(kernel_size, stride, padding);
        m_layers.push_back(pooling2d);

        return pooling2d;
    }
};