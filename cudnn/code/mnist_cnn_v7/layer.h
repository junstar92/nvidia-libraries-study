#pragma once
#include <cudnn.h>
#include <utils.h>

class Layer
{
protected:
    const float m_one = 1.f;
    const float m_zero = 0.f;
    Dims4 m_output_dims{};

public:
    virtual void init(cudnnHandle_t handle, Dims4 input_dims, cudnnTensorFormat_t format, cudnnDataType_t data_type) = 0;
    virtual void forward(cudnnHandle_t handle, const void* src, void* dst) = 0;
    virtual Dims4 getOutputDimension() { return m_output_dims; }
};

class Conv2d : public Layer
{
public:
    Conv2d(int in_channel, int out_channel, int kernel_size, int stride = 1, int padding = 0, int dilation = 1, int groups = 1, bool bias = true, int activation = -1)
     : m_in_channel(in_channel), m_out_channel(out_channel), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation), m_groups(groups), m_bias(bias)
    {
        CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
        CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));
        CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&m_bias_desc));
        CUDNN_ERROR_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
        CUDNN_ERROR_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc));
        CUDNN_ERROR_CHECK(cudnnCreateActivationDescriptor(&m_activation_desc));
        if (activation == -1) {
            m_activation = false;
        }
        else {
            m_activation = true;
            m_activation_mode = static_cast<cudnnActivationMode_t>(activation);
        }
    }
    virtual ~Conv2d() {
        CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(m_input_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(m_output_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(m_bias_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyConvolutionDescriptor(m_conv_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyFilterDescriptor(m_filter_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyActivationDescriptor(m_activation_desc));

        if (m_workspace_dev != nullptr) cudaFree(m_workspace_dev);
    }
    
    void setMode(cudnnConvolutionMode_t mode) {
        m_conv_mode = mode;
    }
    void setAlgo(cudnnConvolutionFwdAlgo_t algo) {
        m_conv_algo = algo;
    }
    void getParameters(void* weight_dev, void* bias_dev = nullptr) {
        m_weight_dev = weight_dev;
        if (m_bias) m_bias_dev = bias_dev;
    }
    virtual void init(cudnnHandle_t handle, Dims4 input_dims, cudnnTensorFormat_t format, cudnnDataType_t data_type) override {
        if (m_kernel_size == 1) {
            input_dims = {input_dims.n, input_dims.c * input_dims.h * input_dims.w, 1, 1};
        }
        CUDNN_ERROR_CHECK(cudnnSetTensor4dDescriptor(m_input_desc, format, data_type, input_dims.n, input_dims.c, input_dims.h, input_dims.w));
        CUDNN_ERROR_CHECK(cudnnSetConvolution2dDescriptor(m_conv_desc, m_padding, m_padding, m_stride, m_stride, m_dilation, m_dilation, m_conv_mode, data_type));
        CUDNN_ERROR_CHECK(cudnnSetConvolutionGroupCount(m_conv_desc, m_groups));
        CUDNN_ERROR_CHECK(cudnnSetConvolutionMathType(m_conv_desc, cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
        CUDNN_ERROR_CHECK(cudnnSetFilter4dDescriptor(m_filter_desc, data_type, format, m_out_channel, m_in_channel, m_kernel_size, m_kernel_size));
        if (m_bias) CUDNN_ERROR_CHECK(cudnnSetTensor4dDescriptor(m_bias_desc, format, data_type, 1, m_out_channel, 1, 1));

        int output_dims[4];
        CUDNN_ERROR_CHECK(cudnnGetConvolutionNdForwardOutputDim(m_conv_desc, m_input_desc, m_filter_desc, 4, output_dims));
        CUDNN_ERROR_CHECK(cudnnSetTensor4dDescriptor(m_output_desc, format, data_type, output_dims[0], output_dims[1], output_dims[2], output_dims[3]));
        m_output_dims = {output_dims[0], output_dims[1], output_dims[2], output_dims[3]};

        if (m_conv_algo == -1) {
            int requested_algo_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
            int returned_algo_count = -1;
            cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            CUDNN_ERROR_CHECK(cudnnFindConvolutionForwardAlgorithm(handle, m_input_desc, m_filter_desc, m_conv_desc, m_output_desc, requested_algo_count, &returned_algo_count, results));
            
            m_conv_algo = results[0].algo;
            m_workspace_size = results[0].memory;
            if (m_workspace_size > 0) {
                CUDA_ERROR_CHECK(cudaMalloc(&m_workspace_dev, m_workspace_size));
            }
        }

        if (m_activation) {
            CUDNN_ERROR_CHECK(cudnnSetActivationDescriptor(m_activation_desc, m_activation_mode, CUDNN_PROPAGATE_NAN, 0.0));
        }
    }
    virtual void forward(cudnnHandle_t handle, const void* src, void* dst) override {
        if (m_bias == true && m_activation == true) {
            CUDNN_ERROR_CHECK(cudnnConvolutionBiasActivationForward(
                handle,
                static_cast<const void*>(&m_one),
                m_input_desc, src,
                m_filter_desc, m_weight_dev,
                m_conv_desc, static_cast<cudnnConvolutionFwdAlgo_t>(m_conv_algo), m_workspace_dev, m_workspace_size,
                static_cast<const void*>(&m_zero),
                m_output_desc, dst,
                m_bias_desc, m_bias_dev,
                m_activation_desc,
                m_output_desc, dst
            ));
        }
        else {
            CUDNN_ERROR_CHECK(cudnnConvolutionForward(
                handle,
                static_cast<const void*>(&m_one),
                m_input_desc, src,
                m_filter_desc, m_weight_dev,
                m_conv_desc, static_cast<cudnnConvolutionFwdAlgo_t>(m_conv_algo), m_workspace_dev, m_workspace_size,
                static_cast<const void*>(&m_zero),
                m_output_desc, dst
            ));
            if (m_bias) {
                CUDNN_ERROR_CHECK(cudnnAddTensor(
                    handle,
                    static_cast<const void*>(&m_one),
                    m_bias_desc, m_bias_dev,
                    static_cast<const void*>(&m_one),
                    m_output_desc, dst
                ));
            }
            if (m_activation) {
                CUDNN_ERROR_CHECK(cudnnActivationForward(
                    handle,
                    m_activation_desc,
                    static_cast<const void*>(&m_one),
                    m_output_desc, dst,
                    static_cast<const void*>(&m_zero),
                    m_output_desc, dst
                ));
            }
        }
    }

private:
    int m_in_channel;
    int m_out_channel;
    int m_kernel_size;
    int m_stride{1};
    int m_padding{0};
    int m_dilation{1};
    int m_groups{1};
    bool m_bias{true};

    cudnnTensorDescriptor_t m_input_desc, m_output_desc, m_bias_desc;
    cudnnConvolutionDescriptor_t m_conv_desc;
    cudnnFilterDescriptor_t m_filter_desc;
    cudnnConvolutionMode_t m_conv_mode{CUDNN_CROSS_CORRELATION};
    bool m_activation{false};
    cudnnActivationDescriptor_t m_activation_desc;
    cudnnActivationMode_t m_activation_mode;
    int m_conv_algo{-1};
    size_t m_workspace_size{0};

    void* m_workspace_dev{nullptr};
    void* m_weight_dev{nullptr};
    void* m_bias_dev{nullptr};
};

class Pooling2d : public Layer
{
public:
    Pooling2d(int kernel_size, int stride, int padding = 0)
     : m_kernel_size(kernel_size), m_stride(stride), m_padding(padding)
    {
        CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
        CUDNN_ERROR_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));
        CUDNN_ERROR_CHECK(cudnnCreatePoolingDescriptor(&m_pooling_desc));
    }
    virtual ~Pooling2d() {
        CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(m_input_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyTensorDescriptor(m_output_desc));
        CUDNN_ERROR_CHECK(cudnnDestroyPoolingDescriptor(m_pooling_desc));
    }

    void setMode(cudnnPoolingMode_t mode) {
        m_pooling_mode = mode;
    }

    virtual void init(cudnnHandle_t handle, Dims4 input_dims, cudnnTensorFormat_t format, cudnnDataType_t data_type) override {
        CUDNN_ERROR_CHECK(cudnnSetTensor4dDescriptor(m_input_desc, format, data_type, input_dims.n, input_dims.c, input_dims.h, input_dims.w));
        CUDNN_ERROR_CHECK(cudnnSetPooling2dDescriptor(m_pooling_desc, m_pooling_mode, CUDNN_PROPAGATE_NAN, m_kernel_size, m_kernel_size, m_padding, m_padding, m_stride, m_stride));

        int output_dims[4];
        CUDNN_ERROR_CHECK(cudnnGetPoolingNdForwardOutputDim(m_pooling_desc, m_input_desc, 4, output_dims));
        CUDNN_ERROR_CHECK(cudnnSetTensor4dDescriptor(m_output_desc, format, data_type, output_dims[0], output_dims[1], output_dims[2], output_dims[3]));
        m_output_dims = {output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    }
    
    virtual void forward(cudnnHandle_t handle, const void* src, void* dst) override {
        CUDNN_ERROR_CHECK(cudnnPoolingForward(
            handle,
            m_pooling_desc,
            static_cast<const void*>(&m_one),
            m_input_desc, src,
            static_cast<const void*>(&m_zero),
            m_output_desc, dst
        ));
    }

private:
    int m_kernel_size;
    int m_stride;
    int m_padding{0};

    cudnnTensorDescriptor_t m_input_desc, m_output_desc;
    cudnnPoolingDescriptor_t m_pooling_desc;
    cudnnPoolingMode_t m_pooling_mode{CUDNN_POOLING_MAX};
};