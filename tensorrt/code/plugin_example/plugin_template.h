#include <NvInfer.h>
#include <string>
#include <vector>

class TemplatePlugin;
class TemplatePluginCreator;

class TemplatePlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    TemplatePlugin() = delete;
    TemplatePlugin(std::string const& name);
    TemplatePlugin(std::string const& name, void const* serialData, const size_t length);

    /**** virtual functions from nvinfer1::IPluginV2 ****/
    // Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
    char const* getPluginType() const noexcept override;
    // Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    char const* getPluginVersion() const noexcept override;
    // Get the number of outputs from the layer.
    int32_t getNbOutputs() const noexcept override;
    // Initialize the layer for execution. This is called when the engine is created.
    int32_t initialize() noexcept override;
    // Release resources acquired during plugin layer initialization. This is called when the engine is
    // destroyed.
    void terminate() noexcept override;
    // Find the size of the serialization buffer required.
    size_t getSerializationSize() const noexcept override;
    // Serialize the layer.
    void serialize(void* buffer) const noexcept override;
    // Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
    void destroy() noexcept override;
    // Set the namespace that this plugin object belongs to. Ideally, all plugin
    // objects from the same plugin library should have the same namespace.
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    // Return the namespace of the plugin object.
    char const* getPluginNamespace() const noexcept override;

    /**** virtual functions from nvifer1::IPluginV2Ext ****/
    // Return the DataType of the plugin output at the requested index.
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    
    /**** virtual functions from nvifer1::IPluginV2Ext ****/
    // Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
    // these parameters.
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    // Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    // Return true if plugin supports the format and datatype for the input/output indexed by pos.
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    // Configure the plugin.
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    // Find the workspace size required by the layer.
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    // Execute the layer.
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    const std::string m_layername;
    std::string m_namespace;
};

class TemplatePluginCreator : public nvinfer1::IPluginCreator
{
public:
    TemplatePluginCreator();
    /**** virtual functions from nvinfer1::IPluginCreator ****/
    // Return the plugin name.
    char const* getPluginName() const noexcept override;
    // Return the plugin version.
    char const* getPluginVersion() const noexcept override;
    // Return a list of fields that needs to be passed to createPlugin.
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    // Return a plugin object. Return nullptr in case of error.
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    // Called during deserialization of plugin layer. Return a plugin object.
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
    // Set the namespace of the plugin creator based on the plugin
    // library it belongs to. This can be set while registering the plugin creator.
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    // Return the namespace of the plugin creator object.
    char const* getPluginNamespace() const noexcept;

private:
    inline static nvinfer1::PluginFieldCollection m_fc{};
    inline static std::vector<nvinfer1::PluginField> m_plugin_attributes{};
    std::string m_namespace;
};