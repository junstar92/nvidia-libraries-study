# Table of Contents

- [Table of Contents](#table-of-contents)
- [Plugins](#plugins)
- [Example](#example)
  - [IPluginV2DynamicExt](#ipluginv2dynamicext)
  - [IPluginCreator](#iplugincreator)
  - [플러그인 레이어 추가](#플러그인-레이어-추가)
  - [전체 구현](#전체-구현)
- [Plugin 함수 호출 순서](#plugin-함수-호출-순서)
  - [NetworkDefinition 생성 시](#networkdefinition-생성-시)
  - [Build Phase](#build-phase)
  - [Execution Phase](#execution-phase)
- [References](#references)

<br>

# Plugins

TensorRT에서 딥러닝 연산을 위한 다양한 레이어들을 지원하지만, 내장된 레이어들로는 부족할 때가 있다. 이런 경우에 플러그인을 사용하여 커스텀 레이어를 구현하여 사용할 수 있다.

플러그인을 사용하려면 `libnvinfer_plugin.so (nvinfer_plugin.dll)` 라이브러리를 로드해야 하며, TensorRT에 내장된 플러그인들은 `initLibNvInferPlugins()`를 먼저 호출해야 사용할 수 있다.

C++ API에서는 아래와 같은 플러그인의 베이스 클래스들을 제공한다.

||Introduced in TensorRT version?|Mixed I/O formats/types|Dynamic Shapes?|Supports implicit/explicit batch mode?|
|--|--|--|--|--|
|[`IPluginV2Ext`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html)|5.1 (deprecated since TensorRT 8.5)|Limited|No|Both|
|[`IPluginV2IOExt`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html)|6.0.1|General|No|Both|
|[`IPluginV2DynamicExt`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html)|6.0.1|General|Yes|Explicit batch mode only|

커스텀 레이어는 이 베이스 클래스들을 상속받아 구현하면 된다. 이번 포스팅에서는 dynamic shape를 지원하는 `IPluginV2DynamicExt`를 중심으로 살펴볼텐데, 다른 클래스들도 비슷하기 때문에 이것으로 커버가 될 것으로 생각된다.

# Example

> 전체 구현은 [link](/tensorrt/code/plugin_example/)에서 확인할 수 있다.

Development Guide 문서에서 예제로 설명하고 있는 플러그인을 직접 구현해보자. 이 플러그인의 요구 사항은 다음과 같다.

- 2 inputs, 2 outputs
- 첫 번째 output은 첫 번째 입력의 copy
- 두 번째 output은 첫 번째 차원에 대한 두 입력 concatenation이다
- 모든 입출력의 타입은 동일하며, 포맷 또한 linear로 동일하다

플러그인을 구현하려면 두 개의 클래스를 구현해야 한다.

- `IPluginV2DynamicExt`를 상속받아 구현하는 플러그인 클래스
- `IPluginCreator`를 상속받아 구현하는 팩토리 클래스 (위 플러그인 클래스 인스턴스를 생성해주는 역할)

각각 상속받는 클래스의 virtual function을 오버라이딩하여 구현해주면 된다.

## IPluginV2DynamicExt

먼저 `IPluginV2DynamicExt`를 상속받아 구현하는 플러그인 클래스부터 살펴보자. 기본적인 선언은 다음과 같다.
```c++
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
```

`IPluginV2DynamicExt` 또한 상속받는 클래스가 있다. 따라서, 구현해야 할 클래스는 아래의 베이스 클래스들의 가상 함수들을 오버라이딩해야 한다.

- `IPluginV2`
- `IPluginV2Ext`
- `IPluginV2DynamicExt`

그럼 위의 클래스 선언에서 구현해야 하는 함수들을 순서대로 살펴보자.

### Class Constructor

먼저 클래스 생성자부터 구현한다. 예제 구현에서는 2개의 생성자를 선언헀으며, 기본 생성자는 삭제하였다. `std::string` 타입 하나를 받는 생성자가 기본적으로 사용되며, 만약 플러그인을 생성할 때 파라미터로 받아야 할 값들이 있다면 함수의 파라미터를 추가해주면 된다. 두 번째 생성자는 Creator 클래스에서 플러그인 인스턴스를 생성할 때 전달되는 인자들로 구성된 생성자이다. 일반적으로 이를 직접 사용하는 경우는 거의 없다고 생각한다.

아래서 serialization data에 대해서 언급하겠지만 예제 구현에서는 플러그인 레이어를 플랜 파일에 저장할 때 별도로 저장이 필요한 데이터가 없으므로 구현이 비어있다.

```c++
TemplatePlugin::TemplatePlugin(std::string const& name)
  : m_layername(name)
{
}

TemplatePlugin::TemplatePlugin(std::string const& name, void const* serialData, const size_t length)
  : m_layername(name)
{
}
```

### `getPluginType()`, `getPluginVersion()`

이 함수들은 해당 플러그인의 타입(이름)과 버전을 반환하는 함수이다. 일반적으로 구현 코드 내에 상수 값으로 설정해두고 이를 그대로 반환해주면 된다.
```c++
namespace {
    const char *PLUGIN_VERSION{"1"};
    const char *PLUGIN_NAME{"TemplatePlugin"};
}

char const* TemplatePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

char const* TemplatePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}
```

### `getNbOutputs()`

이 함수는 플러그인 레이어에서 결과로 출력되는 텐서의 갯수를 반환한다. 예제에서는 2개의 출력이 있으므로 여기서는 2를 리턴하게 된다. 
```c++
//! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
//! prior to any call to initialize().
int32_t TemplatePlugin::getNbOutputs() const noexcept
{
    return 2;
}
```

### `initialize()`, `terminate()`

`initialize()` 함수는 실행을 위해 레이어를 초기화하고, `terminate()` 함수는 엔진이 파괴될 때 호출된다. `initialize()`는 엔진이 생성될 때 호출되는데, 플러그인 구현에 필요한 메모리 할당이 있다면 이 함수에서 수행하면 된다. 메모리 누수를 방지하기 위해 할당된 메모리는 `terminate()` 함수에서 수행해주어야 한다. 예제 구현에서는 별도의 초기화가 필요없기 때문에 두 함수 모두 아무런 동작도 수행하지 않는다.

```c++
//! \brief Initialize the layer for execution. This is called when the engine is created.
//! \return 0 for success, else non-zero (which will cause engine termination).
int32_t TemplatePlugin::initialize() noexcept
{
    return 0;
}

//! \brief Release resources acquired during plugin layer initialization. This is called when the engine is
//! destroyed.
void TemplatePlugin::terminate() noexcept
{
}
```

### `getSerializationSize()`, `serialize()`

`getSerializationSize()` 함수는 컴파일된 엔진이 serialization될 때, 플러그인 레이어에서 serialization되는 데이터의 크기를 반환한다. 그리고 `serialize()`는 인자로 전달된 `buffer` 주소에 serialization할 데이터를 저장한다. 이때 저장되는 크기는 `getSerializationSize()`에서 반환되는 크기와 동일해야 한다. 예제 구현의 경우에서는 별도로 저장해야할 데이터가 없기 때문에 구현은 비어있다.
```c++
//! \brief Find the size of the serialization buffer required.
//! \return The size of the serialization buffer.
size_t TemplatePlugin::getSerializationSize() const noexcept
{
    return 0;
}

//! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by
//! getSerializationSize.
void TemplatePlugin::serialize(void* buffer) const noexcept
{
}
```

만약, 플랜 파일로 serialization해야 할 플러그인 데이터가 있고, 이 데이터가 `int32_t` 타입의 변수 두 개라면, 다음과 같이 구현할 수 있다.
```c++
size_t TemplatePlugin::getSerializationSize() const noexcept
{
    return sizeof(serial_var1) + sizeof(serial_var2);
}
void TemplatePlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer), *a = d;
    auto write = [](char*& buffer, const auto& val) {
        memcpy(buffer, &val, sizeof(val));
        buffer += sizeof(val);
    };

    write(d, serial_var1);
    write(d, serial_var2);

    assert(d == a + getSerializationSize());
}
```

### `destroy()`

`destroy()` 함수는 network or builder or engine이 파괴될 때 호출되며, 플러그인 객체를 파괴한다.

```c++
//! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
void TemplatePlugin::destroy() noexcept
{
}
```

### `setPluginNamespace()`, `getPluginNamespace()`

`setPluginNamespace()`, `getPluginNamespace()` 함수는 생성된 플러그인의 네임스페이스를 설정 및 반환한다.

```c++
//! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
//! objects from the same plugin library should have the same namespace.
//!
//! \param pluginNamespace The namespace for the plugin object.
//! \warning The string pluginNamespace must be 1024 bytes or less including the NULL terminator and must be NULL
//! terminated.
void TemplatePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    m_namespace = std::string(pluginNamespace);
}

//! \brief Return the namespace of the plugin object.
char const* TemplatePlugin::getPluginNamespace() const noexcept
{
    return m_namespace.c_str();
}
```

일반적으로 멤버 데이터 변수로 namespace를 저장하도록 구현한다.

### `getOutputDataType()`

이 함수는 호출될 때 주어진 인덱스의 출력 텐서의 데이터 타입을 반환한다. 즉, 각 출력 텐서의 가능한 데이터 타입을 지정하는 것이다. 함수의 첫 번째 파라미터가 바로 호출 시 전달되는 출력 텐서의 인덱스이다.

예제 구현에서 출력은 두 개이므로 가능한 `index`는 0과 1이다. 단, 구현 요구사항에서 모든 텐서의 타입은 동일하다고 했으므로 입력 텐서의 타입과 동일하도록 맞춰주면 된다. 따라서 `index`와 무관하게 첫 번째 입력 텐서의 데이터 타입으로 반환해준다.
```c++
//! \brief Return the DataType of the plugin output at the requested index.
//!
//! The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no
//! inputs. The returned data type must have a format that is supported by the plugin.
//!
//! \warning DataType:kBOOL not supported.
DataType TemplatePlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}
```

### `clone()`

현재 플러그인 객체를 복사하는 함수이다. 따라서, 만약 플러그인 객체 내부에 데이터가 따로 있다면, 이러한 데이터들도 새로운 객체에 복사해주어야 한다. 복사 생서자(copy constructor) 또는 위에서 구현한 첫 번째 생성자를 사용하여 생성하면 된다. 복사 생성자는 따로 구현하지 않았으므로 다음과 같이 구현하면 충분하다.

```c++
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
```

설명에 따르면 플러그인 내에서 공유되면서 변경이 없는 리소스들은 공유할 수 있다고 언급하고 있다. 따라서, 플로그인 내에서 변경되지 않으면서 모든 플러그인에서 사용되는 리소스들은 static으로 선언하면 같이 사용하도록 하면 불필요한 복사를 피할 수 있다 (e.g., weights).

### ```getOutputDimensions()```

이 함수는 각 출력 텐서의 차원의 expression을 반환한다. Dynamic shape를 사용하는 플러그인이므로 충분히 입력 텐서의 차원에 따라서 출력 텐서의 차원이 변경될 수 있다. 출력 텐서의 차원은 파라미터로 전달되는 입력 텐서의 차원으로부터 계산하여 반환해주면 된다.

예제 구현에서 첫 번째 출력 텐서는 두 번째 입력 텐서의 복사본이므로 두 번째 입력 텐서와 동일한 차원을 갖는다. 두 번째 출력 텐서는 첫 번째 텐서와 두 번째 텐서를 첫 번째 차원으로 concatenation한 것이며 아래와 같이 구현되어 반환된다.

```c++
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
```

### ```supportsFormatCombination()```

`supportsFormatCombination()` 함수는 `pos`로 인덱싱되는 입력/출력에 대해 지원하는 format과 data type을 플러그인이 지원하는지 여부를 반환한다. 입력/출력 텐서는 0부터 `(nbInputs + nbOutputs - 1)`까지 인덱싱되며, 입력 텐서 0부터 `(nbInputs-1)`, 출력 텐서는 `nbInputs`부터 `(nbInputs + nbOutputs - 1)`이다.

어느 한 입출력 텐서의 format과 data type이 다른 텐서에 의존적일 수 있는데, 이 함수는 `pos`번째 텐서를 조사할 때 `inOut[0...pos]`의 조건부로 결과를 반환할 수 있다. `pos`번째 텐서를 조사할 때, `inOut[pos+1...nbInputs + nbOutputs - 1]`는 유효하지 않은 값을 가지므로 이 값들을 이용하면 절대 안된다.

예제 구현에서는 FP32와 FP16을 지원하도록 구현했다.

```c++
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
bool TemplatePlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
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
```

### `configurePlugin()`

`configurePlugin()`은 빌드와 실행 단계에서 여러번 호출될 수 있다. 빌드 단계에서는 `initialize()`가 호출되기 전에 호출되며 `IBuilder`에서 엔진을 생성하는 동안에만 호출된다. 실행 단계에서는 `initialize()`가 호출된 이후에 호출되며 `IBuilder`에 의한 엔진 생성 시와 `IExecutionContext`에 의한 엔진 실행 중에 호출된다.

빌드 단계에서는 프로파일링을 준비하지만 구체적인 입력 크기에 대해서는 준비하지 않을 때 호출된다. 여기서는 플러그인에서 가능한 차원의 범위와 입출력 타입을 기반으로 알고리즘을 선택하는 등의 작업을 수행하면 된다. 인자로 전달되는 `DynamicPluginTensorDesc`는 플러그인이 프로파일링되는 현재 profile의 `kMIN`, `kMAX` 값을 가지고 있다. 또한 이 변수의 `dims` 필드는 네트워크 생성 시 지정된 플러그인의 차원에 해당하며 dynamic shape로 지정된 경우 `-1`을 포함할 수 있다.

실행 단계에서는 지정된 차원의 플러그인을 실행하기 위해서 플러그인을 준비할 때 호출된다. 이때 플러그인에 전달된 `DynamicPluginTensorDesc`의 `dims` 필드에는 `-1`을 포함하지 않는 명시적인 입력 차원이므로 이를 기반으로 알고리즘을 변경하거나 플러그인을 구성할 수 있다.

예제 구현에서는 딱히 구현할 내용이 없기 때문에 빈 구현으로 남겨둔다.
```c++
//! \brief Configure the plugin.
//!
//! configurePlugin() can be called multiple times in both the build and execution phases. The build phase happens
//! before initialize() is called and only occurs during creation of an engine by IBuilder. The execution phase
//! happens after initialize() is called and occurs during both creation of an engine by IBuilder and execution
//! of an engine by IExecutionContext.
//!
//! \param in The input tensors attributes that are used for configuration.
//! \param nbInputs Number of input tensors.
//! \param out The output tensors attributes that are used for configuration.
//! \param nbOutputs Number of output tensors.
//!
void TemplatePlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}
```

### `getWorkspaceSize()`

이 함수는 플러그인 레이어에서 요구하는 추가 workspace 크기를 반환한다. TensorRT 내부에서 직접 할당하여 제공하는 메모리로 확인되며 예제 구현에서는 별도의 메모리가 필요하지 않기 때문에 0을 반환한다.

```c++
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
    return 0;
}
```

### `enqueue()`

레이어를 실행할 때 호출되는 함수이다. 예제 구현에서 첫 번째 출력 텐서는 두 번째 입력 텐서의 단순 복사이고, 두 번째 출력 텐서는 두 입력 텐서의 contenation이다. 따라서, 단순 메모리 복사로 구현이 가능하다. 만약 복잡한 알고리즘을 적용해야 한다면 CUDA 커널 함수를 직접 구현하여 커널 함수가 호출되도록 구현해주어야 한다.

FP32와 FP16 타입을 모두 지원하기 위해 아래와 같이 구현할 수 있다.

```c++
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
```


## IPluginCreator

이번에는 위에서 구현한 플러그인을 위한 Creator 클래스를 구현한다. TensorRT에서 커스텀 레이어를 사용할 때, 이 클래스를 플러그인 레지스트리에 등록해주어야 한다.

`IPluginCreator`를 상속받아 구현되는 클래스 선언은 다음과 같다.
```c++
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
```

### `getPluginName()`, `getPluginVersion()`

이 클래스가 생성하는 플러그인의 이름 및 버전을 반환하는 함수이다. 반환되는 값은 소스 코드 내 상수로 저장하는 값을 사용한다.
```c++
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
```

### `getFieldNames()`

플러그인을 생성하는데 필요한 파라미터 정보를 담고 있는 `PluginFieldCollection` 객체 반환하는 함수이다. `NetworkDefinition`을 사용하여 네트워크를 구성하는 경우에는 필요한 파라미터를 직접 설정하여 전달하면 되겠지만, ONNX와 같은 파일을 파싱하여 네트워크를 구성하는 경우에는 공통의 구조를 가지고 있어야 하므로 `PluginFieldCollection`과 같은 객체가 필요한 것으로 생각된다.

일반적으로 단순히 멤버 변수로 가지고 있는 `PluginFieldCollection` 객체 주소를 반환해주면 된다. 이 객체가 담고 있는 파라미터 리스트 정보는 일반적으로 이 클래스의 생성자에서 설정해준다.

```c++
//! \brief Return a list of fields that needs to be passed to createPlugin.
PluginFieldCollection const* TemplatePluginCreator::getFieldNames() noexcept
{
    return &m_fc;
}
```

### Constructor

일반적으로 생성자에서는 플러그인 생성할 때 필요한 파라미터 정보를 담고 있는 `PluginFieldCollection` 객체와 `PluginField` 객체 리스트를 초기화해준다. 두 객체는 클래스의 멤버 변수로 가지고 있는데, 이 변수들은 변경되지 않고 공용으로 사용되어 일반적으로 static 멤버로 선언된다.

예제 구현에서는 별도의 파라미터 필드가 없어서 아래와 같이 구현된다.
```c++
TemplatePluginCreator::TemplatePluginCreator()
{
    m_plugin_attributes.clear();
    m_fc.nbFields = 0;
    m_fc.fields = nullptr;
}
```

만약 레이어가 네트워크 내 클래스 갯수를 파라미터로 필요로 한다면 다음과 같이 구현할 수 있을 것이다.
```c++
TemplatePluginCreator::TemplatePluginCreator()
{
    m_plugin_attributes.clear();
    m_plugin_attributes.emplace_back("numClasses", nullptr, PluginFieldType::kINT32, 1);
    m_fc.nbFields = m_plugin_attributes.size();
    m_fc.fields = m_plugin_attributes.data();
}
```

### `createPlugin()`

플러그인을 생성하는 함수이다. 호출하는 측에서 각 파라미터의 필드가 채워진 `PluginFieldCollection` 객체를 전달하고, 함수 내에서 이를 이용하여 필요한 파라미터와 함께 플러그인을 생성하여 반환해주면 된다.

예제 구현에서는 필요한 파라미터가 없기 때문에 구현은 다음과 같다.
```c++
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
```

만약 위에서와 같이 클래스 갯수가 파라미터로 필요한 플러그인이라면 다음과 같이 구현할 수 있다.
```c++
IPluginV2* TemplatePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try {
        int32_t num_classes;
        for (int32_t i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name == "numClasses") {
                num_classes = *(static_cast<int32_t const*>(fc->fields[i].data));
            }
        }
        auto* plugin = new TemplatePlugin(name, num_classes);
        plugin->setPluginNamespace(m_namespace.c_str());

        return plugin;
    }
    catch (std::exception const& e) {
        std::clog << e.what() << std::endl;
    }
    return nullptr;
}
```

### `desrializePlugin()`

플랜 파일을 deserialization할 때 플러그인 레이어 객체를 생성하기 위해 호출되는 함수이다. 일반적으로 플러그인을 구현할 때, `deserializePlugin()` 함수에 전달되는 인자 리스트와 동일한 생성자를 구현하고 이를 그대로 사용하여 호출하면 된다.

```c++
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
```

### `getPluginNamespace()`, `setPluginNamespace()`

플러그인의 네임스페이스를 설정 및 반환하는 함수이다. 일반적으로 데이터 멤버로 네임스페이스를 저장한다.

```c++
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
```

## 플러그인 레이어 추가

이렇게 구현한 플러그인은 `INetworkDefinition`를 통해 네트워크에 추가하거나 ONNX 등의 파일을 파싱하면서 연산과 일치하는 플러그인을 추가하게 된다. 아래 코드는 `INetworkDefinition`을 사용하여 네트워크를 구성할 때, 플러그인 레이어를 추가하는 방법을 보여준다.
```c++
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
```

## 전체 구현

예제 플러그인과 해당 플러그인 레이어만으로 구성된 네트워크를 빌드하고 실제 실행까지 해보는 코드는 아래에서 확인할 수 있다.

- [main.cpp](/tensorrt/code/plugin_example/main.cpp)
- [plugin_template.h](/tensorrt/code/plugin_example/plugin_template.h)
- [plugin_template.cpp](/tensorrt/code/plugin_example/plugin_template.cpp)

`main.cpp` 코드를 컴파일하고 실행하면 다음의 결과를 확인할 수 있다. 인자로 1을 전달하면 `kFP16` 플래그가 활성화되어 빌드 시 FP16 구현을 후보로 둔다. 주의해야 할 점은 입출력 텐서의 타입을 따로 지정하지 않으면 FP16 구현을 활성화하더라도 입출력 텐서의 타입은 FP16이 아니라는 점이다. 다만, `main.cpp` 구현에서는 입출력 텐서의 타입을 configuration에 설정한 타입과 동일하도록 지정하였기 때문에 컴파일되는 엔진의 입출력 텐서 또한 설정된 타입을 따라가게 된다.
```
$ ./main
Type: FP16
Creating builder...
[I] [MemUsageChange] Init CUDA: CPU +329, GPU +0, now: CPU 334, GPU 1228 (MiB)
[I] [MemUsageChange] Init builder kernel library: CPU +327, GPU +104, now: CPU 680, GPU 1332 (MiB)
Creating network...
[W] Tensor DataType is determined at build time for tensors not marked as input or output.
[W] Tensor DataType is determined at build time for tensors not marked as input or output.
Setting build configuration...
Building serialized network...
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +856, GPU +360, now: CPU 1536, GPU 1692 (MiB)
[I] [MemUsageChange] Init cuDNN: CPU +125, GPU +58, now: CPU 1661, GPU 1750 (MiB)
[I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[I] Detected 2 inputs and 2 output network tensors.
[I] Total Host Persistent Memory: 224
[I] Total Device Persistent Memory: 0
[I] Total Scratch Memory: 0
[I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 4 MiB
[I] Total Activation Memory: 0
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 1661, GPU 1758 (MiB)
[I] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 1662, GPU 1768 (MiB)
[I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
[W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 1661, GPU 1734 (MiB)
[I] Loaded engine size: 0 MiB
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 1661, GPU 1744 (MiB)
[I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1661, GPU 1752 (MiB)
[I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 1661, GPU 1744 (MiB)
[I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1661, GPU 1752 (MiB)
[I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
> Input 1 ([1,2,2])
[[[  72.438   32.250 ]
  [  77.625   82.625 ]]]
> Input 2 ([1,2,2])
[[[  96.438    9.562 ]
  [  47.094   70.188 ]]]

> Output 3 ([1,2,2])
[[[  96.438    9.562 ]
  [  47.094   70.188 ]]]
> Output 4 ([1,4,2])
[[[  72.438   32.250 ]
  [  77.625   82.625 ]
  [  96.438    9.562 ]
  [  47.094   70.188 ]]]
```

마지막 입출력 텐서의 값을 확인해보면 첫 번째 출력 텐서는 단순히 두 번째 입력 텐서의 복사이고, 두 번째 출력 텐서는 두 입력 텐서의 concat 연산 결과라는 것을 확인할 수 있다.

# Plugin 함수 호출 순서

오버라이딩한 플러그인 함수 구현이 빌드 및 실행 시 어떠한 순서로 호출되는지 궁금해서 프린트를 찍어서 호출 순서를 살펴보았다.

## NetworkDefinition 생성 시

우선 `INetworkDefinition`으로 네트워크를 구성할 때 호출되는 순서는 다음과 같다.

1. creator로 인한 plugin 인스턴스 생성
   1. default constructor
   2. `setPluginNamespace()` in default constructor
2. `network->addPluginV2()`를 통해 플러그인 레이어를 `INetworkDefinition`에 추가
   1. `getNbOutputs()`
   2. `getOutputDataType()` for output 0
   3. `getOutputDataType()` for output 1
   4. `clone()`
      1. default constructor
      2. `setPluginNamespace()`
      3. `getPluginType()`

Creator를 통해 플러그인을 생성하는 중에는 당연히 기본 생성자에서 `setPluginNamespace()`를 호출하는 것 밖에 없다.

그 이후에 생성된 플러그인 객체를 전달하면서 네트워크 레이어로 추가할 때, 출력의 갯수를 확인하기 위해 `getNbOutputs()`를 호출하고 각 output 텐서의 타입을 쿼리한다. 그리고 `clone()`을 통해 최종적으로 네트워크에 추가하는 것으로 추측된다.

## Build Phase

빌드하는 과정에서는 여러 함수들이 다양하게 많이 호출된다. 호출 과정은 다음과 같다.
```
Building serialized network...
> calling getOutputDataType()
> calling getOutputDataType()
> calling getOutputDimensions()
> calling getNbOutputs()
> calling getOutputDimensions()
> calling getNbOutputs()
> calling clone()
> calling default constructor
> calling setPluginNamespace()
> calling getOutputDimensions()
> calling getNbOutputs()
> calling getOutputDimensions()
> calling getNbOutputs()
[V] Applying generic optimizations to the graph for inference.
[V] Original: 1 layers
[V] After dead-layer removal: 1 layers
> calling getPluginType()
[V] After Myelin optimization: 1 layers
[V] Applying ScaleNodes fusions.
[V] After scale fusion: 1 layers
[V] After dupe layer removal: 1 layers
[V] After final dead-layer removal: 1 layers
[V] After tensor merging: 1 layers
[V] After vertical fusions: 1 layers
[V] After dupe layer removal: 1 layers
[V] After final dead-layer removal: 1 layers
[V] After tensor merging: 1 layers
[V] After slice removal: 1 layers
[V] After concat removal: 1 layers
[V] Trying to split Reshape and strided tensor
[V] Graph construction and optimization completed in 0.000545952 seconds.
[V] Using cublasLt as a tactic source
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +856, GPU +360, now: CPU 1536, GPU 1694 (MiB)
[V] Using cuDNN as a tactic source
[I] [MemUsageChange] Init cuDNN: CPU +125, GPU +58, now: CPU 1661, GPU 1752 (MiB)
[I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[V] Constructing optimization profile number 0 [1/1].
> calling clone()
> calling default constructor
> calling setPluginNamespace()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling supportsFormatCombination()
[V] Reserving memory for host IO tensors. Host: 0 bytes
[V] =============== Computing reformatting costs
[V] =============== Computing reformatting costs
[V] =============== Computing reformatting costs
[V] =============== Computing reformatting costs
[V] =============== Computing costs for 
[V] *************** Autotuning format combination: Float(4,2,1), Float(4,2,1) -> Float(4,2,1), Float(8,2,1) ***************
[V] Formats and tactics selection completed in 8.9389e-05 seconds.
[V] After reformat layers: 1 layers
[V] Pre-optimized block assignment.
[V] Block size 10469376000
[V] Total Activation Memory: 10469376000
[I] Detected 2 inputs and 2 output network tensors.
> calling clone()
> calling default constructor
> calling setPluginNamespace()
> calling configurePlugin()
> calling getWorkspaceSize()
> calling getWorkspaceSize()
> calling getWorkspaceSize()
> calling getWorkspaceSize()
[V] Layer: (Unnamed Layer* 0) [PluginV2DynamicExt] Host Persistent: 224 Device Persistent: 0 Scratch Memory: 0
[I] Total Host Persistent Memory: 224
[I] Total Device Persistent Memory: 0
[I] Total Scratch Memory: 0
[I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 0 MiB
[V] Optimized block assignment.
[I] Total Activation Memory: 0
> calling destroy()
[V] Disabling unused tactic source: EDGE_MASK_CONVOLUTIONS
[V] Using cublasLt as a tactic source
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 1661, GPU 1760 (MiB)
[V] Using cuDNN as a tactic source
[I] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 1662, GPU 1770 (MiB)
> calling initialize()
[V] Engine generation completed in 0.52149 seconds.
[V] Engine Layer Information:
Layer(PluginV2): (Unnamed Layer* 0) [PluginV2DynamicExt], Tactic: 0x0000000000000000, input1[Float(1,2,2)], input2[Float(1,2,2)] -> (Unnamed Layer* 0) [PluginV2DynamicExt]_output_0[Float(1,2,2)], (Unnamed Layer* 0) [PluginV2DynamicExt]_output_1[Float(1,4,2)]
> calling destroy()
[I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
[W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
> calling getPluginType()
> calling getPluginVersion()
> calling getPluginNamespace()
> calling getSerializationSize()
> calling serialize()
[W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
> calling getPluginType()
> calling getPluginVersion()
> calling getPluginNamespace()
> calling getSerializationSize()
> calling serialize()
> calling terminate()
> calling destroy()
```

초반부에서는 네트워크 전체를 구성하는 과정에서 호출되는 것으로 추측된다. 그런 다음 `supportsFormatCombination()`을 호출하면서 가능한 조합을 수집하고 이를 바탕으로 timing을 수행한다. FP16이 설정된 경우에는 더욱 많은 `supportsFormatCombination()`이 호출되었다. FP32의 경우에는 가능한 조합이 하나뿐이기 때문에 별다른 timing 측정이 발생하지 않은 것으로 확인된다. FP16으로 설정한 경우에는 아래와 같이 `supportsCombination()` 호출 이후에 timing 측정이 발생한다.

```
...
> calling supportsFormatCombination()
> calling supportsFormatCombination()
[V] Reserving memory for host IO tensors. Host: 0 bytes
[V] =============== Computing reformatting costs
[V] *************** Autotuning Reformat: Half(4,2,1) -> Float(4,2,1) ***************
[V] --------------- Timing Runner: Optimizer Reformat(input1 -> <out>) (Reformat)
[V] Tactic: 0x00000000000003e8 Time: 0.00272477
[V] Tactic: 0x00000000000003ea Time: 0.00476968
[V] Tactic: 0x0000000000000000 Time: 0.00323118
[V] Fastest Tactic: 0x00000000000003e8 Time: 0.00272477
[V] =============== Computing reformatting costs
[V] *************** Autotuning Reformat: Half(4,2,1) -> Float(4,2,1) ***************
[V] =============== Computing reformatting costs
[V] *************** Autotuning Reformat: Float(4,2,1) -> Half(4,2,1) ***************
[V] --------------- Timing Runner: Optimizer Reformat(<in> -> (Unnamed Layer* 0) [PluginV2DynamicExt]_output_0) (Reformat)
[V] Tactic: 0x00000000000003e8 Time: 0.00343423
[V] Tactic: 0x00000000000003ea Time: 0.0053523
[V] Tactic: 0x0000000000000000 Time: 0.00404832
[V] Fastest Tactic: 0x00000000000003e8 Time: 0.00343423
[V] =============== Computing reformatting costs
[V] *************** Autotuning Reformat: Float(8,2,1) -> Half(8,2,1) ***************
[V] --------------- Timing Runner: Optimizer Reformat(<in> -> (Unnamed Layer* 0) [PluginV2DynamicExt]_output_1) (Reformat)
[V] Tactic: 0x00000000000003e8 Time: 0.00302395
[V] Tactic: 0x00000000000003ea Time: 0.00529117
[V] Tactic: 0x0000000000000000 Time: 0.00330107
[V] Fastest Tactic: 0x00000000000003e8 Time: 0.00302395
[V] =============== Computing costs for 
[V] *************** Autotuning format combination: Float(4,2,1), Float(4,2,1) -> Float(4,2,1), Float(8,2,1) ***************
> calling supportsFormatCombination()
> calling supportsFormatCombination()
...
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling clone()
> calling default constructor
> calling setPluginNamespace()
[V] --------------- Timing Runner: (Unnamed Layer* 0) [PluginV2DynamicExt] (PluginV2)
> calling configurePlugin()
> calling getWorkspaceSize()
> calling getWorkspaceSize()
> calling initialize()
> calling configurePlugin()
> calling enqueue()
> calling enqueue()
...
> calling enqueue()
> calling enqueue()
[V] Tactic: 0x0000000000000000 Time: 0.00828722
> calling terminate()
[V] Fastest Tactic: 0x0000000000000000 Time: 0.00828722
[V] >>>>>>>>>>>>>>> Chose Runner Type: PluginV2 Tactic: 0x0000000000000000
> calling destroy()
[V] *************** Autotuning format combination: Half(4,2,1), Half(4,2,1) -> Half(4,2,1), Half(8,2,1) ***************
> calling supportsFormatCombination()
> calling supportsFormatCombination()
...
> calling supportsFormatCombination()
> calling supportsFormatCombination()
> calling clone()
> calling default constructor
> calling setPluginNamespace()
[V] --------------- Timing Runner: (Unnamed Layer* 0) [PluginV2DynamicExt] (PluginV2)
> calling configurePlugin()
> calling getWorkspaceSize()
> calling getWorkspaceSize()
> calling initialize()
> calling configurePlugin()
> calling enqueue()
> calling enqueue()
...
> calling enqueue()
> calling enqueue()
[V] Tactic: 0x0000000000000000 Time: 0.00834159
> calling terminate()
[V] Fastest Tactic: 0x0000000000000000 Time: 0.00834159
[V] >>>>>>>>>>>>>>> Chose Runner Type: PluginV2 Tactic: 0x0000000000000000
> calling destroy()
```
위 로그를 살펴보면 Float -> Float 연산과 Half -> Half 연산의 속도를 측정하는 것을 확인할 수 있다. 이 과정에서 하나의 포맷 조합에 대해서 아래의 호출 과정이 확인된다.

- `supportsFormatCombination()` (several times)
- `clone()`
- `configurePlugin()`
- `getWorkspaceSize()` x 2
- `initialize()`
- `configurePlugin()`
- `enqueue()` (several times)
- `terminate()`
- `destroy()`

> FP16이 활성화된 빌드 로그를 살펴보면 FP32의 경우가 속도가 더 빠른 것으로 측정된다. 입출력의 크기가 작아서 두 정밀도에서의 복사 속도 차이가 크지 않기 때문이다. 실제로 입력 텐서의 크기를 크게 지정하면 Half -> Half 구현이 더 빠른 것을 확인할 수 있다.
>
> 다만, 엔진의 입출력 텐서 타입을 FP16으로 명시적으로 설정했기 때문에 Float -> Float의 속도가 더 빠르게 측정되었더라도 Half -> Half로 선택되는 것으로 보인다.

## Execution Phase

실행 과정에서부터 종료까지의 로그는 다음과 같다.
```
Creating runtime...
[I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 1661, GPU 1736 (MiB)
Creating deserialized engine...
[I] Loaded engine size: 0 MiB
> calling default constructor for deserialization
> calling setPluginNamespace()
[V] Using cublasLt as a tactic source
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 1661, GPU 1746 (MiB)
[V] Using cuDNN as a tactic source
[I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1661, GPU 1754 (MiB)
> calling initialize()
[V] Deserialization required 2468 microseconds.
[I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
Creating execution context...
[V] Using cublasLt as a tactic source
[I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 1661, GPU 1746 (MiB)
[V] Using cuDNN as a tactic source
[I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1661, GPU 1754 (MiB)
> calling clone()
> calling default constructor
> calling setPluginNamespace()
[V] Total per-runner device persistent memory is 0
[V] Total per-runner host persistent memory is 224
[V] Allocated activation device memory of size 0
[I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
Starting inference...
> Input 1 ([1,2,2])
[[[  60.531   57.219 ]
  [   1.538    7.855 ]]]
> Input 2 ([1,2,2])
[[[  36.969   81.812 ]
  [  31.188   71.812 ]]]
> calling configurePlugin()
> calling enqueue()

> Output 3 ([1,2,2])
[[[  36.969   81.812 ]
  [  31.188   71.812 ]]]
> Output 4 ([1,4,2])
[[[  60.531   57.219 ]
  [   1.538    7.855 ]
  [  36.969   81.812 ]
  [  31.188   71.812 ]]]
> calling destroy()
> calling terminate()
> calling destroy()
> calling destroy()
```

Deserialization 과정에서는 Creator를 통해 deserialization을 위한 생성자가 호출되는 것으로 확인된다. 그런 다음 `initialize()` 함수가 호출되어 네트워크 초기화를 마치는 것으로 보인다.

그리고 `IExecutioContext`를 생성하는 과정에서 `clone()` 함수가 호출된다.

실제 추론 과정에서는 아래의 두 함수가 호출된다.

1. `configurePlugin()`
2. `enqueue()`

확인 결과, 첫 `executeV2()` 호출에서만 `configurePlugin()`이 호출되며, 연속된 `executeV2()` 호출에서는 `enqueue()`만 호출된다. 만약 dynamic shape를 사용하여 입력의 차원이 변경된 이후에는 `configurePlugin()`이 다시 호출될 것이다.

# References

- [Extending TensorRT with Custom Layers](/tensorrt/doc/01_developer_guide/09_extending_tensorrt_with_custom_layers.md)
- [NvInferPlugin.h Reference](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/_nv_infer_plugin_8h.html)