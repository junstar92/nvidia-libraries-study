//! Compile Command:
//! $ nvcc -I. -I/usr/local/tensorrt/include -I/usr/local/cuda/include -L/usr/local/tensorrt/lib -lnvinfer -lnvinfer_plugin -o main main.cpp plugin_template.cpp

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <plugin_template.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <iomanip>

using namespace nvinfer1;

class Logger : public ILogger
{
    using Severity = ILogger::Severity;
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

void createNetwork(INetworkDefinition* network, const DataType type);
template<typename T, size_t N = sizeof(T), typename D = float>
void infer(ICudaEngine& engine, IExecutionContext& context);

int main(int argc, char** argv)
{
    DataType type = DataType::kFLOAT;
    if (argc > 1) {
        type = static_cast<DataType>(std::stoi(argv[1]));
        if (type > DataType::kHALF) return 1;
    }
    std::cout << "Type: " << ((type == DataType::kFLOAT) ? "FP32" : "FP16") << std::endl;
    // create builder
    std::cout << "Creating builder...\n";
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    
    // create network
    std::cout << "Creating network...\n";
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flag));
    createNetwork(network.get(), type);

    // builder configuration
    std::cout << "Setting build configuration...\n";
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (type == DataType::kHALF) config->setFlag(BuilderFlag::kFP16);
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
    
    // build serialized network
    std::cout << "Building serialized network...\n";
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    // create engine
    std::cout << "Creating runtime...\n";
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    std::cout << "Creating deserialized engine...\n";
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

    // inference
    std::cout << "Creating execution context...\n";
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    std::cout << "Starting inference...\n";
    if (type == DataType::kFLOAT) {
        infer<float>(*engine, *context);
    }
    else if (type == DataType::kHALF) {
        infer<half>(*engine, *context);
    }

    return 0;
}

void createNetwork(INetworkDefinition* network, const DataType type)
{
    // register plugin by using macro defined in TensorRT header
    REGISTER_TENSORRT_PLUGIN(TemplatePluginCreator);

    // add inputs
    auto input1 = network->addInput("input1", type, Dims{3, 1, 2, 2});
    auto input2 = network->addInput("input2", type, Dims{3, 1, 2, 2});

    // add plugin layer
    auto creator = getPluginRegistry()->getPluginCreator("TemplatePlugin", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    PluginFieldCollection* fc = new PluginFieldCollection();
    fc->nbFields = pluginData->nbFields;
    fc->fields = nullptr;
    std::cout << "creatPlugin\n";
    IPluginV2* plugin_obj = creator->createPlugin("template plugin", fc);
    ITensor* tensors[] = { input1, input2 };
    std::cout << "addPluginV2\n";
    auto plugin = network->addPluginV2(tensors, 2, *plugin_obj);
    std::cout << "addPluginV2 end\n";
    plugin->getOutput(0)->setType(type);
    plugin->getOutput(1)->setType(type);

    // mark output
    network->markOutput(*plugin->getOutput(0));
    network->markOutput(*plugin->getOutput(1));

    delete fc;
}

template<typename T, size_t N = sizeof(T), typename D = float>
void infer(ICudaEngine& engine, IExecutionContext& context)
{
    // memory allocation
    const auto nbBinding = engine.getNbBindings();
    std::vector<Dims> dims(4);
    for (int i = 0; i < nbBinding; i++) {
        dims[i] = engine.getBindingDimensions(i);
    }
    std::vector<T*> host_buffers(4);
    std::vector<void*> dev_buffers(4);
    for (int i = 0; i < nbBinding; i++) {
        size_t count = std::accumulate(dims[i].d, dims[i].d + dims[i].nbDims, 1, std::multiplies<>());
        host_buffers[i] = new T[count];
        cudaMalloc(&dev_buffers[i], count * N);
    }

    // memory random init
    std::random_device rd;
    std::mt19937 e(rd());
    std::uniform_real_distribution<> generator(0, 99);
    if constexpr (N == 4) {
        for (int i = 0; i < nbBinding; i++) {
            size_t count = std::accumulate(dims[i].d, dims[i].d + dims[i].nbDims, 1, std::multiplies<>());
            for (size_t j = 0; j < count; j++) {
                host_buffers[i][j] = generator(e);
            }
            cudaMemcpy(dev_buffers[i], host_buffers[i], count * N, cudaMemcpyHostToDevice);
        }
    }
    else {
        for (int i = 0; i < nbBinding; i++) {
            size_t count = std::accumulate(dims[i].d, dims[i].d + dims[i].nbDims, 1, std::multiplies<>());
            for (size_t j = 0; j < count; j++) {
                host_buffers[i][j] = __float2half(generator(e));
            }
            cudaMemcpy(dev_buffers[i], host_buffers[i], count * N, cudaMemcpyHostToDevice);
        }
    }

    // display input data
    auto dimsToStr = [](Dims dim) {
        std::string ret = "[";
        for (int i = 0; i < dim.nbDims; i++) {
            ret += std::to_string(dim.d[i]) + ",";
        }
        ret[ret.length() - 1] = ']';
        return ret;
    };
    for (int i = 0; i < 2; i++) {
        int sp = 2;
        std::cout << "> Input " << i + 1 << " (" << dimsToStr(dims[i]) << ")" << std::endl;
        std::cout << "[";
        for (int n = 0; n < dims[i].d[0]; n++) {
            std::cout << "[";
            for (int h = 0; h < dims[i].d[1]; h++) {
                if (h != 0)
                    std::cout << std::setw(sp) << " ";
                std::cout << "[";
                for (int w = 0; w < dims[i].d[2]; w++) {
                    int index = (n * dims[i].d[1] * dims[i].d[2]) + (h * dims[i].d[2]) + w;
                    if constexpr (N == 4) {
                        std::cout << std::fixed << std::setprecision(3) << std::setw(8) << host_buffers[i][index] << " ";
                    }
                    else {
                        std::cout << std::fixed << std::setprecision(3) << std::setw(8) << __half2float(host_buffers[i][index]) << " ";
                    }
                }
                std::cout << "]";
                if (h != dims[i].d[1] - 1) std::cout << std::endl;
            }
            std::cout << "]";
        }
        std::cout << "]\n";
    }

    // run inference
    context.executeV2(dev_buffers.data());
    cudaDeviceSynchronize();

    // copy result from device to host
    for (int i = 2; i < nbBinding; i++) {
        size_t count = std::accumulate(dims[i].d, dims[i].d + dims[i].nbDims, 1, std::multiplies<>());
        cudaMemcpy(host_buffers[i], dev_buffers[i], count * N, cudaMemcpyDeviceToHost);
    }

    // print output data
    std::cout << std::endl;
    for (int i = 2; i < 4; i++) {
        int sp = 2;
        std::cout << "> Output " << i + 1 << " (" << dimsToStr(dims[i]) << ")" << std::endl;
        std::cout << "[";
        for (int n = 0; n < dims[i].d[0]; n++) {
            std::cout << "[";
            for (int h = 0; h < dims[i].d[1]; h++) {
                if (h != 0)
                    std::cout << std::setw(sp) << " ";
                std::cout << "[";
                for (int w = 0; w < dims[i].d[2]; w++) {
                    int index = (n * dims[i].d[1] * dims[i].d[2]) + (h * dims[i].d[2]) + w;
                    if constexpr (N == 4) {
                        std::cout << std::fixed << std::setprecision(3) << std::setw(8) << host_buffers[i][index] << " ";
                    }
                    else {
                        std::cout << std::fixed << std::setprecision(3) << std::setw(8) << __half2float(host_buffers[i][index]) << " ";
                    }
                }
                std::cout << "]";
                if (h != dims[i].d[1] - 1) std::cout << std::endl;
            }
            std::cout << "]";
        }
        std::cout << "]\n";
    }

    // free memory
    for (int i = 0; i < nbBinding; i++) {
        delete[] host_buffers[i];
        cudaFree(dev_buffers[i]);
    }
}