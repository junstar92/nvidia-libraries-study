#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <iterator>
#include <numeric>

using namespace nvinfer1;
using namespace nvonnxparser;

static const std::string CONV1_WEIGHT = "params/conv1_weight.bin";
static const std::string CONV1_BIAS = "params/conv1_bias.bin";
static const std::string CONV2_WEIGHT = "params/conv2_weight.bin";
static const std::string CONV2_BIAS = "params/conv2_bias.bin";
static const std::string FC_WEIGHT = "params/fc_weight.bin";
static const std::string FC_BIAS = "params/fc_bias.bin";

class Logger : public nvinfer1::ILogger
{
    using Severity = nvinfer1::ILogger::Severity;
public:
    Logger(Severity severity = Severity::kERROR) : mSeverity{severity} {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= mSeverity) {
            printf("%s %s\n",
                levelPrefix(severity), msg
            );
        }
    }

private:
    Severity mSeverity{Severity::kERROR};

    const char* levelPrefix(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "[F]";
            case Severity::kERROR: return "[E]";
            case Severity::kWARNING: return "[W]";
            case Severity::kINFO: return "[I]";
            case Severity::kVERBOSE: return "[V]";
            default: return "";
        }
        if (severity == Severity::kINTERNAL_ERROR) {
            return "F";
        }
    }
} gLogger(ILogger::Severity::kVERBOSE);

void *conv1_weight, *conv1_bias, *conv2_weight, *conv2_bias, *fc_weight, *fc_bias;

void loadBinary(void* ptr, const int count, const char* filename)
{
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(ptr), count * 4); // support only fp32
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
    // load weights in host memory
    conv1_weight = (void*)malloc(32 * 3 * 3 * sizeof(float));
    conv1_bias = (void*)malloc(32 * sizeof(float));
    conv2_weight = (void*)malloc(64 * 32 * 3 * 3 * sizeof(float));
    conv2_bias = (void*)malloc(64 * sizeof(float));
    fc_weight = (void*)malloc(7 * 7 * 64 * 10 * sizeof(float));
    fc_bias = (void*)malloc(10 * sizeof(float));

    loadBinary(conv1_weight, 32 * 3 * 3, CONV1_WEIGHT.c_str());
    loadBinary(conv1_bias, 32, CONV1_BIAS.c_str());
    loadBinary(conv2_weight, 64 * 32 * 3 * 3, CONV2_WEIGHT.c_str());
    loadBinary(conv2_bias, 64, CONV2_BIAS.c_str());
    loadBinary(fc_weight, 7 * 7 * 64 * 10, FC_WEIGHT.c_str());
    loadBinary(fc_bias, 10, FC_BIAS.c_str());

    std::ifstream ifs("plan.bin");
    /**** Build Phase ****/
    if (!ifs.is_open()) {
        // tensorrt instance
        IBuilder* builder = createInferBuilder(gLogger);

        // create network
        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(flag);

        // create parser
        IParser* parser = createParser(*network, gLogger);

        // parsing onnx file
        parser->parseFromFile("mnist_cnn.onnx", static_cast<int32_t>(ILogger::Severity::kVERBOSE));

        // build configuration
        IBuilderConfig* config = builder->createBuilderConfig();
        config->clearFlag(BuilderFlag::kTF32);
        config->setFlag(BuilderFlag::kFP16);
        
        // build serialized network
        auto plan = builder->buildSerializedNetwork(*network, *config);

        // save serialized engine as plan in memory
        std::ofstream ofs("plan.bin"); 
        ofs.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        
        // delete instances
        delete network;
        delete parser;
        delete config;
        delete builder;
        delete plan;
    }
    ifs.close();

    /**** Runtime Phase ****/
    ifs.open("plan.bin", std::ios_base::binary);
    ifs >> std::noskipws;
    std::string plan;
    std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), std::back_inserter(plan));

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());

    // memory allocation for input, output
    Dims input_dims = engine->getBindingDimensions(0);
    Dims output_dims = engine->getBindingDimensions(1);

    void *input, *output;
    void *d_input, *d_output;

    size_t count = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<>());
    input = (void*)malloc(count * sizeof(float));
    cudaMalloc(&d_input, count * sizeof(float));

    count = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<>());
    output = (void*)malloc(count * sizeof(float));
    cudaMalloc(&d_output, count * sizeof(float));

    cudaEvent_t start, stop;
    float msec = 0.f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    IExecutionContext* context = engine->createExecutionContext();
    void* const binding[] = {d_input, d_output};
    for (int i = 0; i < 10; i++) {
        // get input data
        std::string filename = "digits/" + std::to_string(i) + ".bin";
        loadBinary((void*)input, 28 * 28, filename.c_str());
        show_digit((float*)input, 28, 28);
        cudaMemcpy(d_input, input, sizeof(float) * 28 * 28, cudaMemcpyHostToDevice);

        // inference
        cudaEventRecord(start);
        context->executeV2(binding);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msec, start, stop);

        // extract output
        cudaMemcpy(output, d_output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        
        auto iter = std::max_element((float*)output, (float*)output + 10);
        int output_digit = std::distance((float*)output, iter);
        std::cout << "Digit: " << output_digit << " (" << get_prob((float*)output, output_digit) << ")\n";
        std::cout << "Elapsed Time: " << msec << " ms\n\n";
    }

    // free memory & instances
    delete context;
    delete engine;
    delete runtime;

    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    free(conv1_weight);
    free(conv1_bias);
    free(conv2_weight);
    free(conv2_bias);
    free(fc_weight);
    free(fc_bias);

    return 0;
}