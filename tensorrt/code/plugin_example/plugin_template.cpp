#include <plugin_template.h>
#include <iostream>
#include <cassert>
#include <numeric>
#include <cuda_runtime.h>

using namespace nvinfer1;

namespace {
    const char *PLUGIN_VERSION{"1"};
    const char *PLUGIN_NAME{"TemplatePlugin"};
}

TemplatePlugin::TemplatePlugin(std::string const& name)
  : m_layername(name)
{
    // std::cout << "> calling default constructor\n";
}

TemplatePlugin::TemplatePlugin(std::string const& name, void const* serialData, const size_t length)
  : m_layername(name)
{
    // std::cout << "> calling default constructor for deserialization\n";
}

char const* TemplatePlugin::getPluginType() const noexcept
{
    // std::cout << "> calling getPluginType()\n";
    return PLUGIN_NAME;
}

char const* TemplatePlugin::getPluginVersion() const noexcept
{
    // std::cout << "> calling getPluginVersion()\n";
    return PLUGIN_VERSION;
}

//! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
//! prior to any call to initialize().
int32_t TemplatePlugin::getNbOutputs() const noexcept
{
    // std::cout << "> calling getNbOutputs()\n";
    return 2;
}

//! \brief Initialize the layer for execution. This is called when the engine is created.
//! \return 0 for success, else non-zero (which will cause engine termination).
int32_t TemplatePlugin::initialize() noexcept
{
    // std::cout << "> calling initialize()\n";
    return 0;
}

//! \brief Release resources acquired during plugin layer initialization. This is called when the engine is
//! destroyed.
void TemplatePlugin::terminate() noexcept
{
    // std::cout << "> calling terminate()\n";
}

//! \brief Find the size of the serialization buffer required.
//! \return The size of the serialization buffer.
size_t TemplatePlugin::getSerializationSize() const noexcept
{
    // std::cout << "> calling getSerializationSize()\n";
    return 0;
}

//! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by
//! getSerializationSize.
void TemplatePlugin::serialize(void* buffer) const noexcept
{
    // std::cout << "> calling serialize()\n";
}

//! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
void TemplatePlugin::destroy() noexcept
{
    // std::cout << "> calling destroy()\n";
}

//! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
//! objects from the same plugin library should have the same namespace.
//!
//! \param pluginNamespace The namespace for the plugin object.
//! \warning The string pluginNamespace must be 1024 bytes or less including the NULL terminator and must be NULL
//! terminated.
void TemplatePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    // std::cout << "> calling setPluginNamespace()\n";
    m_namespace = std::string(pluginNamespace);
}

//! \brief Return the namespace of the plugin object.
char const* TemplatePlugin::getPluginNamespace() const noexcept
{
    // std::cout << "> calling getPluginNamespace()\n";
    return m_namespace.c_str();
}

//! \brief Return the DataType of the plugin output at the requested index.
//!
//! The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no
//! inputs. The returned data type must have a format that is supported by the plugin.
//!
//! \warning DataType:kBOOL not supported.
DataType TemplatePlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // std::cout << "> calling getOutputDataType()\n";
    return inputTypes[0];
}

//! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
//! these parameters. If the source plugin is pre-configured with configurePlugin(), the returned object
//! should also be pre-configured. The returned object should allow attachToContext() with a new execution context.
//! Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object
//! (e.g. via ref-counting) to avoid duplication.
//!
//! The TensorRT runtime calls clone() to clone the plugin when an execution context is created for an engine,
//! after the engine has been created.  The runtime does not call initialize() on the cloned plugin,
//! so the cloned plugin should be created in an initialized state.
IPluginV2DynamicExt* TemplatePlugin::clone() const noexcept
{
    // std::cout << "> calling clone()\n";
    try {
        auto* plugin = new TemplatePlugin(this->m_layername);
        plugin->setPluginNamespace(this->m_namespace.c_str());
        
        return plugin;
    }
    catch (const std::exception& e) {
        std::clog << e.what() << std::endl;
    }
    return nullptr;
}


//! \brief Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.
//!
//! \param outputIndex The index of the output tensor
//! \param inputs Expressions for dimensions of the input tensors
//! \param nbInputs The number of input tensors
//! \param exprBuilder Object for generating new expressions
//!
//! This function is called by the implementations of IBuilder during analysis of the network.
//!
//! Example #1: A plugin has a single output that transposes the last two dimensions of the plugin's single input.
//! The body of the override of getOutputDimensions can be:
//!
//!     DimsExprs output(inputs[0]);
//!     std::swap(output.d[output.nbDims-1], output.d[output.nbDims-2]);
//!     return output;
//!
//! Example #2: A plugin concatenates its two inputs along the first dimension.
//! The body of the override of getOutputDimensions can be:
//!
//!     DimsExprs output(inputs[0]);
//!     output.d[0] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[0], *inputs[1].d[0]);
//!     return output;
//!
DimsExprs TemplatePlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // std::cout << "> calling getOutputDimensions()\n";
    assert(nbInputs == 2);
    assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    assert(inputs[0].nbDims >= 2);

    DimsExprs out_dim{};
    if (outputIndex == 0) {
        out_dim = inputs[1];
    }
    else if (outputIndex == 1) {
        out_dim = inputs[0];
        out_dim.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *out_dim.d[1], *inputs[1].d[1]);
    }
    
    return out_dim;
}

//! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
//!
//! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
//! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs-1.
//!
//! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
//! by inOut[pos].format and inOut[pos].type.  The override should return true if that format/datatype at inOut[pos]
//! are supported by the plugin.  If support is conditional on other input/output formats/datatypes, the plugin can
//! make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
//! that the plugin supports.  The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1],
//! which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
//!
//! Some examples:
//!
//! * A definition for a plugin that supports only FP16 NCHW:
//!
//!         return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
//!
//! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
//!   and FP32 NCHW for its single output:
//!
//!         return inOut.format[pos] == TensorFormat::kLINEAR && (inOut.type[pos] == pos < 2 ?  DataType::kHALF :
//!         DataType::kFLOAT);
//!
//! * A definition for a "polymorphic" plugin with two inputs and one output that supports
//!   any format or type, but the inputs and output must have the same format and type:
//!
//!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
//!
//! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
//!
bool TemplatePlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // std::cout << "> calling supportsFormatCombination()\n";
    assert(nbInputs == 2);
    assert(nbOutputs == 2);
    assert(pos >= 0 && pos < 4);

    switch (pos)
    {
    case 0:
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF)
                && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    case 2:
        [[ fallthrough ]];
    case 3:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    }
    
    return false;
}

//! \brief Configure the plugin.
//!
//! configurePlugin() can be called multiple times in both the build and execution phases. The build phase happens
//! before initialize() is called and only occurs during creation of an engine by IBuilder. The execution phase
//! happens after initialize() is called and occurs during both creation of an engine by IBuilder and execution
//! of an engine by IExecutionContext.
//!
//! Build phase:
//! IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for profiling but not for any
//! specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of
//! input and output formats, along with the bound of possible dimensions. The min and max value of the
//! DynamicPluginTensorDesc correspond to the kMIN and kMAX value of the current profile that the plugin is being
//! profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network creation.
//! Wildcard dimensions will exist during this phase in the desc.dims field.
//!
//! Execution phase:
//! IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for executing the plugin for a
//! specific dimensions. This provides an opportunity for the plugin to change algorithmic choices based on the
//! explicit input dimensions stored in desc.dims field.
//!  * IBuilder will call this function once per profile, with desc.dims resolved to the values specified by the
//!  kOPT
//!    field of the current profile. Wildcard dimensions will not exist during this phase.
//!  * IExecutionContext will call this during the next subsequent instance enqueue[V2]() or execute[V2]() if:
//!    - The batch size is changed from previous call of execute()/enqueue() if hasImplicitBatchDimension() returns
//!    true.
//!    - The optimization profile is changed via setOptimizationProfile() or setOptimizationProfileAsync().
//!    - An input shape binding is changed via setInputShapeBinding().
//!    - An input execution binding is changed via setBindingDimensions().
//! \warning The execution phase is timing critical during IExecutionContext but is not part of the timing loop when
//! called from IBuilder. Performance bottlenecks of configurePlugin won't show up during engine building but will
//! be visible during execution after calling functions that trigger layer resource updates.
//!
//! \param in The input tensors attributes that are used for configuration.
//! \param nbInputs Number of input tensors.
//! \param out The output tensors attributes that are used for configuration.
//! \param nbOutputs Number of output tensors.
//!
void TemplatePlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // std::cout << "> calling configurePlugin()\n";
}

//! \brief Find the workspace size required by the layer.
//!
//! This function is called after the plugin is configured, and possibly during execution.
//! The result should be a sufficient workspace size to deal with inputs and outputs of the given size
//! or any smaller problem.
//!
//! \return The workspace size.
size_t TemplatePlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept
{
    // std::cout << "> calling getWorkspaceSize()\n";
    return 0;
}


//! \brief Execute the layer.
//!
//! \param inputDesc how to interpret the memory for the input tensors.
//! \param outputDesc how to interpret the memory for the output tensors.
//! \param inputs The memory for the input tensors.
//! \param outputs The memory for the output tensors.
//! \param workspace Workspace for execution.
//! \param stream The stream in which to execute the kernels.
//!
//! \return 0 for success, else non-zero (which will cause engine termination).
int32_t TemplatePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // std::cout << "> calling enqueue()\n";
    size_t N;
    if (inputDesc[0].type == DataType::kFLOAT) N = 4;
    else N = 2;

    // output 0 is a copy of input 1
    Dims out0_dims = outputDesc[0].dims;
    size_t count = std::accumulate(out0_dims.d, out0_dims.d + out0_dims.nbDims, 1, std::multiplies<>());
    auto err = cudaMemcpy(outputs[0], inputs[1], count * N, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;
    
    // output 1 is a contenation of input 0 and input 1
    err = cudaMemcpy(outputs[1], inputs[0], count * N, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(outputs[1] + count * N, inputs[1], count * N, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;

    return err;
}


TemplatePluginCreator::TemplatePluginCreator()
{
    m_plugin_attributes.clear();
    m_fc.nbFields = 0;
    m_fc.fields = nullptr;
}

//! \brief Return the plugin name.
//!
//! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
//! terminated.
char const* TemplatePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

//! \brief Return the plugin version.
//!
//! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
//! terminated.
char const* TemplatePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

//! \brief Return a list of fields that needs to be passed to createPlugin.
PluginFieldCollection const* TemplatePluginCreator::getFieldNames() noexcept
{
    return &m_fc;
}

//! \brief Return a plugin object. Return nullptr in case of error.
IPluginV2* TemplatePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try {
        auto* plugin = new TemplatePlugin(name);
        plugin->setPluginNamespace(m_namespace.c_str());
        
        return plugin;
    }
    catch (std::exception const& e) {
        std::clog << e.what() << std::endl;
    }
    return nullptr;
}

//! \brief Called during deserialization of plugin layer. Return a plugin object.
IPluginV2* TemplatePluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try {
        auto* plugin = new TemplatePlugin(name, serialData, serialLength);
        plugin->setPluginNamespace(m_namespace.c_str());

        return plugin;
    }
    catch (std::exception const& e) {
        std::clog << e.what() << std::endl;
    }
    return nullptr;
}

//! \brief Set the namespace of the plugin creator based on the plugin
//! library it belongs to. This can be set while registering the plugin creator.
void TemplatePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    m_namespace = std::string(pluginNamespace);
}

//! \brief Return the namespace of the plugin creator object.
//!
//! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
//! terminated.
char const* TemplatePluginCreator::getPluginNamespace() const noexcept
{
    return m_namespace.c_str();
}