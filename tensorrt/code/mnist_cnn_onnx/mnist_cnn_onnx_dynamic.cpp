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
} gLogger(ILogger::Severity::kERROR);

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
        parser->parseFromFile("mnist_cnn_dynamic.onnx", static_cast<int32_t>(ILogger::Severity::kERROR));

        // build configuration
        IBuilderConfig* config = builder->createBuilderConfig();
        config->clearFlag(BuilderFlag::kTF32);
        // config->setFlag(BuilderFlag::kFP16);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);

        // optimization profile
        auto profiler = builder->createOptimizationProfile();
        auto input_dims = network->getInput(0)->getDimensions();
        input_dims.d[0] = 1;
        profiler->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, input_dims);
        input_dims.d[0] = 10;
        profiler->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = 64;
        profiler->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, input_dims);

        config->addOptimizationProfile(profiler);
        
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
    int32_t batch_size = 10;
    Dims input_dims = engine->getBindingDimensions(0);
    Dims output_dims = engine->getBindingDimensions(1);
    printf("> input  dims: (%d, %d, %d, %d)\n", input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    printf("> output dims: (%d, %d)\n", output_dims.d[0], output_dims.d[1]);

    input_dims.d[0] = batch_size;
    output_dims.d[0] = batch_size;

    float *input, *output;
    void *d_input, *d_output;

    size_t in_count = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<>());
    input = (float*)malloc(in_count * sizeof(float));
    cudaMalloc(&d_input, in_count * sizeof(float));

    size_t out_count = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<>());
    output = (float*)malloc(out_count * sizeof(float));
    cudaMalloc(&d_output, out_count * sizeof(float));

    cudaEvent_t start, stop;
    float msec = 0.f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    IExecutionContext* context = engine->createExecutionContext();
    context->setTensorAddress(engine->getBindingName(0), d_input);
    context->setTensorAddress(engine->getBindingName(1), d_output);

    // copy mnist data from host to device
    for (int i = 0; i < 10; i++) {
        std::string filename = "digits/" + std::to_string(i) + ".bin";
        loadBinary((void*)(input + i * 28 * 28), 28 * 28, filename.c_str());
    }
    cudaMemcpy(d_input, input, sizeof(float) * in_count, cudaMemcpyHostToDevice);

    // inference
    context->setBindingDimensions(0, input_dims);
    if (!context->allInputShapesSpecified()) {
        printf("> Input dimension is not specified\n");
    }
    else {
        cudaEventRecord(start);
        context->enqueueV3(nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msec, start, stop);
    }

    // extract output
    cudaMemcpy(output, d_output, sizeof(float) * out_count, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 10; i++) {
        auto iter = std::max_element((float*)(output + i * 10), (float*)(output + (i + 1) * 10));
        int output_digit = std::distance((float*)(output + i * 10), iter);
        std::cout << "Digit: " << output_digit << " (" << get_prob((float*)(output + i * 10), output_digit) << ")\n";
    }
    std::cout << "Elapsed Time: " << msec << " ms\n\n";

    // free memory & instances
    delete context;
    delete engine;
    delete runtime;

    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}