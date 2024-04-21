# Table of Contents

- [Table of Contents](#table-of-contents)
- [Extending TensorRT with Custome Layers](#extending-tensorrt-with-custome-layers)
- [Adding Custom Layers Using the C++ API](#adding-custom-layers-using-the-c-api)
- [Adding Custom Layers Using the Python API](#adding-custom-layers-using-the-python-api)
- [Enabling Timing Caching and Using Custom Tactics](#enabling-timing-caching-and-using-custom-tactics)
- [Sharing Custom Resources Among Plugins](#sharing-custom-resources-among-plugins)
- [Using Custom Layers When Importing a Model with a Parser](#using-custom-layers-when-importing-a-model-with-a-parser)
- [Plugin API Description](#plugin-api-description)
- [Migrating V2 Plugins to IPluginV3](#migraing-older-v2-plugins-to-ipluginv3)
- [Coding Guidelines for Plugins](#coding-guidelines-for-plugins)
- [Plugin Shared Libraries](#plugin-shared-libraries)
- [References](#references)

> [TensorRT 10.0] `IPluginV3`가 새롭게 도입됨

# Extending TensorRT with Custome Layers

TensorRT는 다양한 레이어를 지원하고 기능은 지속적으로 확장된다. 그러나 지원되는 레이어가 특정 요구 사항을 만족하지 못하는 경우가 있을 수 있다. 이런 경우, TensorRT는 **플러그인(plugin)** 이라는 커스텀 레이어를 구현하여 확장할 수 있다.

TensorRT에는 사용할 수 있는 몇 가지 플러그인이 이미 포함되어 있으며, 포함되어 있는 플러그인은 [GitHub: TensorRT plulgins](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins)에서 확인할 수 있다.

어플리케이션에서 TensorRT 플러그인을 사용하려면 `libinfer_plugin.so` (`nvinfer_plugin.dll` for windows) 라이브러리가 반드시 로드되어야 한다. 그리고 어플리케이션 코드에서 `initLibNvInferPlugin`을 호출하여 모든 플러그인을 등록해야 한다.

당연히 자신만의 플러그인(커스텀 레이어)를 구현하고 사용할 수도 있다.

# Adding Custom Layers Using the C++ API

플러그인을 구현 및 사용하는 순서는 다음과 같다.

1. TensorRT 플러그인의 base classes 중 하나에서 파생하여 플러그인 클래스를 구현한다. 현재 `IPluginV3`을 권장한다.
2. TensorRT 플러그인 creator 클래스 중 하나에서 파생하여 플러그인 클래스와 연결된 plugin creator class를 구현한다. 현재 `IPluginCreatorV3One`을 권장한다.
3. TensorRT의 plugin registry에 plugin creator class의 인스턴스를 등록한다.
4. TensorRT의 network APIs를 직접 사용하거나, TensorRT ONNX parser APIs를 통해 ONNX 모델을 로드하여 TensorRT 네트워크에 플러그인 클래스의 인스턴스를 추가한다.

아래에서 각 단계의 세부 내용을 살펴볼 수 있다.

## Implementing a Plugin Class

커스텀 레이어는 TensorRT 플러그인의 베이스 클래스 중 하나를 상속받아 구현할 수 있다.

TensorRT 10.0부터 권장되는 유일한 plugin interface는 `IPluginV3`이며, 다른 인터페이스는 삭제 예정이다. 아래 내용은 `IPluginV3`를 구현하는 방법에 대한 내용이 대부분이며, V2 plugin interfaces를 `IPluginV3`로 마이그레이션하는 방법은 [Migrating V2 Plugins to IPluginV3]()에서 확인할 수 있다.

`IPluginV3`는 core, build, runtime이라는 3가지 기능을 정의하는 _capability interfaces_에 대한 wrapper이다.

- **Core capability** : refers to plugin attributes and behaviors common to both build and runtime phases of a plugin's lifetime
- **Build capability** : refers to plugin attributes and behaviors that the plugin must exhibit for the TensorRT builder
- **Runtime capability** : refers to plugin attributes and behaviors that the plugin must exhibit for it to be executable, either during auto-tuning in the TensorRT build phase or inference in the TensorRT runtime phase

[`IPluginV3OneCore`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/class_i_plugin_v3_one_core.html), [`IPluginV3OneBuild`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/class_i_plugin_v3_one_build.html), [`IPluginV3OneRuntime`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/class_i_plugin_v3_one_runtime.html)는 각각 core, build, runtime capabilities를 표시하기 위해  `IPluginV3` 플러그인으로 구현해야 하는 base classes이다.

## Implementing a Plugin Creator Class

네트워크에 플러그인을 사용하기 위해서는 먼저 `PluginRegistry`에 이를 등록해주어야 한다. 플러그인을 직접 등록하는 대신, 해당 플러그인에 대한 팩토리 클래스(plugin creator)의 인스턴스를 등록해주면 된다. 팩토리 클래스는 `IPluginCreatorInterface`의 child class를 파생하여 구현된다. Plugin creator는 플러그인의 name, version, plugin field parameters에 대한 정보도 가지고 있다.

`IPluginCreatorV3One`은 `IPluginV3`의 팩토리 클래스이다. 즉, `IPluginCreatorV3One::createPlugin()`은 `IPluginV3` 타입의 플러그인 객체를 반환하며, 이 함수의 signature는 다음과 같다.
```c++
IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase phase)
```

이 함수는 build phase 또는 runtime phase에서 플러그인 인스턴스를 생성할 때 호출될 수 있고, `TensorRTPhase` 타입인 `phase` 인자로 어떤 phase인지 전달된다.

- In both phase, the returned `IPluginV3` object must have a valid core capability.
- In the build phase, the returned `IPluginV3` object must have both a build and runtime capability.
- In the runtime phase, the returned `IPluginV3` object must have a runtime capability. A build capability is not required, and is ignored.

## Registering a Plugin Creator with the Plugin Registry

Registry에 플러그인을 등록하는 방법에는 두 가지가 있다.

- `REGISTER_TENSORRT_PLUGIN` 매크로를 사용하여 정적으로 plugin creator를 등록할 수 있다. 이 경우에는 항상 default namespace("")으로 creator가 등록된다.
- `initLibNvInferPlugins`와 유사한 자체 entry-point를 생성하고 plugin registry에서 `registerCreator`를 호출하여 동적으로 등록할 수 있다. 이 방법으로는 고유한 namespace로 등록할 수 있다. Namespace를 통해 서로 다른 플러그인 라이브러리에서 빌드할 때 충돌을 피할 수 있다.

Serialization하는 동안, TensorRT engine은 내부적으로 모든 플러그인들의 name, version, namespace과 `IPluginV3OneRuntime::getFieldsToSerialize()`에서 반환되는 `PluginFieldCollection` 내의 모든 plugin fields를 저장한다.

Deserialization하는 동안, TensorRT는 plugin registry에서 동일한 plugin name, version 및 namespace를 가진 plugin creator를 찾고, 이에 대해 `IPluginCreatorV3One::createPlugin()`을 호출한다. 직렬화된 `PluginFieldCollection`은 `fc` argument로 다시 전달된다.

## Adding a Plugin Instance to a TensorRT Network

`addPluginV3()`를 사용하여 네트워크에 plugin을 추가할 수 있다.

```c++
// Look up the plugin in the registry
// Cast to appropriate child class of IPluginCreatorInterface
auto creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(pluginName, pluginVersion, pluginNamespace));
PluginFieldCollection const* pluginFC = creator->getFieldNames();
// Populate the fields parameters for the plugin layer 
// PluginFieldCollection *pluginData = parseAndFillFields(pluginFC, layerFields); 
// Create the plugin object using the layerName and the plugin meta data, for use by the TensorRT builder
IPluginV3 *pluginObj = creator->createPlugin(layerName, pluginData, TensorRTPhase::kBUILD);
// Add the plugin to the TensorRT network 
auto layer = network.addPluginV3(inputs.data(), int(inputs.size()),  shapeInputs.data(), int(shapeInputs.size()), pluginObj);
... (build rest of the network and serialize engine)
// Delete the plugin object
delete pluginObj;
... (free allocated pluginData)
```

> **Note**: `createPlugin`는 새로운 plugin 객체를 heap에 생성하여 객체의 포인터를 반환한다. 위 예제 코드처럼 memory leak을 피하기 위해 `pluginObj`를 직접 제거해주어야 한다.
>
> 엔진이 제거될 때, engine이 빌드되는 동안 생성된 plugin object의 복사본들은 engine에 의해서 제거된다. 처음 네트워크에 추가할 때 생성한 plugin 객체만 사용자가 확실히 제거해주면 된다.

## Example: Adding a Custom Layer with Dynamic Shape Support Using C++

이미지의 크기가 네트워크에 들어가기 전에 32 x 32로 reshape되는 상황에서 사용할 수 있는 padding-like operation을 커스텀 레이어로 구현하는 경우를 살펴보자. 즉, (B, C, H, W) 차원의 입력 텐서 `X`를 받아서 (B, C, 32, 32) 차원의 출력 텐서 `Y`를 생성하는 레이어(`PadPlugin`)이다.

`IPluginV3` 플러그인은 별도의 인터페이스로 정의된 여러 capabilities를 보유해야 하므로, 다중 상속을 사용하여 플러그인을 구현할 수 있다.

다중 상속을 이용하여 `PadPlugin`은 다음과 같이 구현할 수 있다.
```c++
class PadPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
    ... override inherited virtual methods.
};
```

`IPluginV3::getCapabilityInterface`의 override는 반드시 개별 capability interface에 대한 포인터를 반환해야 한다. 각 `PluginCapabilityType`에 대한 해당 capability interface로 캐스팅하여 컴파일러의 모호성을 제거하는 것이 필수적이다.
```c++
IPluginCapability* PadPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept override
{
    try {
        if (type == PluginCapabilityType::kBUILD) {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME) {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch {
        // log error
    }
    return nullptr;
}
```

`PadPlugin`에서 중요한 메소드들은 다음과 같다.

- `INetworkDefinition::addPluginV3`
- `IPluginV3OneBuild::getNbOutputs`
- `IPluginV3OneBuild::getOutputDataTypes`
- `IPluginV3OneBuild::getOutputShapes`
- `IPluginV3OneBuild::supportsFormatCombination`
- `IPluginV3OneBuild::configurePlugin`
- `IPluginV3OneRuntime::onShapeChange`
- `IPluginV3OneRuntime::enqueue`

`INetworkDefinition::addPluginV3`를 사용하여 네트워크에 plugin을 추가할 수 있다.
```c++
std::vector<ITensor*> inputs{X};
auto pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *plugin);
```

Plugin 레이어의 출력이 하나라면 `IPluginV3OneBuild::getNbOutputs`를 통해 이를 지정할 수 있다.
```c++
int32_t PadPlugin::getNbOutputs() const noexcept override
{
    return 1;
}
```

출력 텐서의 데이터 타입은 `IPluginV3OneBuild::getOutputDataTypes`로 알려줄 수 있으며, 입력 타입과 항상 동일하다면 다음과 같이 구현할 수 있다.
```c++
int32_t PadPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
{
    outputTypes[0] = inputTypes[0];
    return 0;
}
```

`getOutputShapes`는 data-dependent output shapes를 제외하고 input 차원 측면에서의 output 차원에 대한 _symbolic expressions_ 를 반환한다. `PadPlugin` 예제에서 output의 처음 두 차원의 값은 input의 처음 두 차원과 동일하고, 나머지 두 차원의 값은 둘 다 32가 된다. 인자로 전달된 `IExprBuilder`를 사용하여 constant symbolic expressions를 정의할 수 있다.
```c++
int32_t PadPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
{
    outputs[0].nbDims = 4;
    // first two output dims are equal to the first two input dims
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[1];
    // the last two output dims are equal to 32
    outputs[0].d[2] = exprBuilder.constant(32)
    outputs[0].d[3] = exprBuilder.constant(32)
    return 0;
}
```

TensorRT는 `supportsFormatCombination`을 사용하여 주어진 타입과 포맷 조합이 가능한 지 쿼리한다. 파라미터로 주어지는 `pos` 인덱스 값은 0부터 시작한다. `PadPlugin` 예제에서 입력 인덱스는 0이고 출력 인덱스는 1이다.
```c++
bool PadPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
{
    assert(0 <= pos && pos < 2);
    return inOut[pos].desc.format == PluginFormat::kLINEAR && inOut[pos].desc.type == DataType::kFLOAT;
}
```

TensorRT는 auto-tuning 중(engine build phase) 및 엔진이 실행되는 중(runtime phase) 모두에서 플러그인이 `enqueue()`를 실행하기 전에 configuration을 선택할 수 있도록 아래의 두 가지 메소드를 호출한다.

- `IPluginV3OneBuild::configurePlugin` : plugin이 profiling(auto-tune)을 준비하면서 특정 input size가 아닌 경우에 호출된다.
- `IPluginV3OneRuntime::onShapeChange` : build-phaes 및 runtime-phase 모두에서 `enqueue()`가 호출되기 전에 실제 input/output shapes를 통신하기 위해 호출된다 (`enqueue()`의 subsequent). output의 `PluginTensorDesc`는 `getOutputShapes()`를 통해 지정된 모든 data-dependent 차원에 대한 wildcards (-1)이 포함된다.

`PadPlugin`은 위의 두 메소드가 필요하지 않아서 아무런 구현도 하지 않아도 된다.
```c++
int32_t PadPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    return 0;
}

int32_t PadPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    return 0;
}
```

마지막으로, `PadPlugin::enqueue`는 실제 작업을 수행한다. Shapes가 dynamic이므로 enqueue에는 입력 및 출력의 실제 크기, 포맷, 타입을 설명하는 `PluginTensorDesc`가 전달된다.
```c++
int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
{
    // populate outputs and return status code	
}
```

## Example: Adding a Custom Layer with a Data-Dependent and Shape Input-Dependent Shapes Using C++

이 예제는 data-dependent shapes와 shape-input dependent shapes를 갖는 플러그인 예제이다. 이 기능은 V3 plugins에서 새로 추가된 기능이다.

- **Data-dependent Shapes (DDS)** : plugin output의 shape가 input tensors의 값에 결정됨
- **Shape inputs** : plugin은 device tensor inputs 이외에도 shape tensor inputs을 받을 수 있다. 이는 plugin에서 `IPluginV3OneBuild::getOutputShapes()`의 인자로만 표시된다. 이 텐서의 유일한 목적인 output shape 계산을 위한 것이다.

하나의 device input `X`, 하나의 shape input `S`, 그리고 output `Y`를 갖는 `BarPlugin`를 예시로 살펴보자.

- The first dimension of `Y` depends on the value of `S`
- The second dimension of `Y` is static
- The third dimension of `Y` is data-dependent
- The fourth dimension of `Y` depends on the shape of `X`

이전 예제인 `PadPlugin`과 유사하게 `BarPlugin`도 다중 상속을 사용한다.

마찬가지로 `INetworkDefinition::addPluginV3`를 사용하여 네트워크에 플러그인 레이어를 추가할 수 있다. 여기서는 이전과 다르게 shape tensor inputs를 위한 추가적인 두 인자가 있으며, device tensor inputs 이후에 위치한다.
```c++
std::vector<ITensor*> inputs{X};
std::vector<ITensor*> shapeInputs{S};
auto pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), shapeInputs.data(), shapeInputs.size(), *plugin);
```

`getOutputShapes`의 override에서 플러그인은 반드시 각 output tensor의 각 data-dependent 차원의 position과 bound를 모두 선언해야 한다. Bound는 size tensor라는 특별한 output으로 표현될 수 있다. Size tensor는 `INT32` 또는 `INT64` 타입의 스칼라이며, auto-tuning을 위한 값과 upper bound를 통해 표현된다. 이 값들은 상수이거나 `IExprBuilder`를 사용하여 device input shapes 또는 shape inputs values로 계산될 수 있다.

`BarPlugin`의 경우, 단일 data-dependent 차원이 있으며, 이는 하나의 size tensor를 사용하여 표현할 수 있다. Data-dependent 차원을 표현하는데 필요한 모든 size tensor는 플러그인의 output이 된다. 따라서, 이 플러그인에는 2개의 output이 있다.
```c++
int32_t BarPlugin::getNbOutputs() const noexcept override
{
    return 2;
}
```

`Y`는 device input `X`와 동일한 타입이고 data-dependent 차원의 크기는 `INT32`에 딱 맞는다고 가정한다. 그러면 output data types는 다음과 같이 표현할 수 있다.
```c++
int32_t BarPlugin::getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
{
    outputTypes[0] = inputTypes[0];
    outputTypes[1] = DataType::kINT32;
    return 0;
}
```

`getOutputShapes`는 전달된 `IExprBuilder`를 사용하여 symbolic output shape expressions를 빌드할 수 있다. Size tensors는 반드시 0-D로 선언되어야 한다.

```c++
int32_t BarPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
{
    outputs[0].nbDims = 4;
    // the first output dimension depends on the value of S
    // the value of S is encoded as fictitious dimensions
    outputs[0].d[0] = shapeInputs[0].d[0];
    // the thrid output dimension depends on the shape of X
    outputs[0].d[2] = inputs[0].d[0];
    // the second output dimension is static
    outputs[0].d[1] = exprBuilder.constant(3);

    auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    auto optValue = exprBuilder.operation(DimenionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));

    // output at index 1 is a size tensor
    outputs[1].nbDims = 0; // size tensors must be declared as 0-D
    auto sizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

    // the fourth output dimension is data-dependent
    outputs[0].d[3] = sizeTensor;

    return 0;
}
```

`supportsFormatCombination`에는 다음의 조건이 적용된다.
- the device input `X` must have `DataType::kFLOAT` or `DataType::kHALF`
- the output `Y` must have the same type as `X`
- the size tensor output has type `DataType::kINT32`

```c++
bool BarPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
{
    assert(0 <= pos && pos < 3);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;

    bool typeOk{false};

    switch (pos)
    {
    case 0: typeOk = in[0].desc.type == DataType::kFLOAT || in[0].desc.type == DataType::kHALF; break;
    case 1: typeOk = out[0].desc.type == in[0].desc.type; break;
    case 2: typeOk = out[1].desc.type == DataType::kINT32; break;
    }
    
    return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
}
```

> `supportsFormatCombination` 구현에서는 전달된 `pos` 값보다 낮은 인덱스의 format/type만을 사용해야 한다.

나머지 구현은 이전의 `PadPlugin`의 구현과 동일하다. 한 가지 주목할 점은 `onShapeChange`에서 output의 `PluginTensorDesc`에는 data-dependent dimension에 대해 wildcard (-1)가 포함된다는 것이다.

Data-dependent output shapes를 가지는 `enqueue` 구현은 static 또는 dynamic shape 경우과 크게 다르지 않다. 다른 output과 마찬가지로 output에 data-dependent dimension을 가지는 경우, `enqueue`에 전달된 output buffer는 대응하는 output tensor를 보유할 수 있을 만큼 충분히 크도록 보장된다. 이 크기는 `getOutputShapes`를 통해 지정된 upper-bound를 기반으로 결정된다.

## Example: Adding a Custom Layer with INT8 I/O Support Using C++

`PoolPlugin`을 통해 어떻게 플러그인에 `INT8` I/O를 추가할 수 있는지 살펴보자. 대부분의 구현은 이전 두 예제와 유사하다.

INT8 I/O에 영향을 미치는 메소드는 다음과 같다.

- `supportsFormatCombination`
- `configurePlugin`

`supportsFormatCombination`은 반드시 허용되는 INT8 I/O 조합을 알려주어야 한다. `PoolPlugin`의 경우, 지원하는 I/O tensor format은 FP32, FP16, BF16, FP8, INT8 data type의 linear CHW이며, I/O tensor는 반드시 동일한 데이터 타입이어야 한다.
```c++
bool PoolPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
{
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].desc.format == PluginFormat::kLINEAR;
    condition &= (inOut[pos].desc.type == DataType::kFLOAT ||
                  inOut[pos].desc.type == DataType::kHALF ||
             inOut[pos].desc.type == DataType::kBF16 ||
                   inOut[pos].desc.type == DataType::kFP8 ||
                  inOut[pos].desc.type == DataType::kINT8);
    condition &= inOut[pos].desc.type == inOut[0].desc.type;
    return condition;
}
```

- If INT8 calibration must be used with a network with INT8 I/O plugins, the plugin must support FP32 I/O as TensorRT uses FP32 to calibrate the graph.
- If the FP32 I/O variant is not supported or INT8 calibration is not used, all required INT8 I/O tensors scales must be set explicitly.
- Calibration cannot determine the dynamic range of a plugin internal tensors. Plugins that operate on quantized data must calculate their own dynamic range for internal tensors.
- A plugin can be designed to accept both FP8 and INT8 I/O types, although note that in TensorRT 9.0 the builder does not alow networks that mix INT8 and FP8.

`configurePlugin` 또는 `onShapeChange`를 통해 TensorRT가 전달하는 정보들은 pooling parameters와 input/output scales에 관련된 정보를 얻는데 사용될 수 있다. 여기서는 이를 멤버 변수로 저장되고 직렬화된 다음, 추론 중에 사용되도록 역직렬화될 수 있다.
```c++
int32_t PoolPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    ...
    mPoolingParams.mC = in.desc.d[1];
    mPoolingParams.mH = in.desc.d[2];
    mPoolingParams.mW = in.desc.d[3];
    mPoolingParams.mP = out.desc.d[2];
    mPoolingParams.mQ = ou.desc.d[3];
    mInHostScale = in[0].desc.scale >= 0.0F ? in[0].desc.scale : -1.0F;
    mOutHostScale = out[0].desc.scale >= 0.0F ? out[0].desc.scale : -1.0F;
}
```

텐서 별 INT8 I/O scales는 `PluginTensorDesc::scale`로부터 얻을 수 있다.

# Adding Custom Layers Using the Python API

파이썬 API에서의 방법은 [문서](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer_python)를 참조

# Enabling Timing Caching and Using Custom Tactics

`IPluginV3`는 V2 plugin 이전에서는 불가능했었던 커스텀 레이어에 대한 profiling을 제어할 수 있는 기능을 제공한다. 그 기능 중 하나는 timing caching을 활성화하는 것이다. TensorRT 네트워크에 동일한 플러그인에 대한 여러 인스턴스가 포함되어 있으며 동일하게 구성되어 있고(예를 들어, 플러그인 속성 값들이 같은 경우), 동일한 입출력 및 타입인 경우에 하나의 인스턴스에 대한 timing을 캐싱하는 것이 합리적이다. 이 기능을 통해 하나의 인스턴스에 대한 latency만 측정하고 캐싱한 뒤, 나머지 인스턴스에 대해서는 이를 사용하여 측정을 건너뛴다.

`IPluginV3`의 timing caching은 opt-in 기능이며, opt-in하려면 플러그인이 null이 아닌 timing cache ID를 알려주어야 한다.
```c++
char const* FooPlugin::getTimingCacheID() noexcept override
{
    // return nullptr to disable timing caching (default behavior)
    // return non-null string to enable timing caching
}
```

Timing caching ID를 사용할 때는 다음 사항에 주의해야 한다.
- 사용자가 제공하는 timing caching ID는 더 큰 timing caching ID에 대한 suffix로 간주되어야 한다. TensorRT는 플러그인의 입출력 형태와 타입 정보를 고려하여 자동으로 suffix를 형성한다. 대부분의 경우, 사용자가 제공한 timing caching ID는 plugin attributes와 이들의 값으로 구성될 수 있다.
- 오직 플러그인의 creation state만 반영해야 한다.

V2 plugin에 대해 TensorRT는 지원하는 모든 타입/포맷 조합에 대해서만 plugin의 timing을 측정한다. `IPluginV3`를 사용하면 custom tactics의 timing을 측정하도록 하는 기능도 있으며, 가장 빠른 tactics가 사용된다. 예를 들어, 플러그인에는 output을 계산하기 위한 두 개의 커널 중 하나가 있을 수 있으며, 특정 플랫폼과 특정 입출력 포맷 및 타입에 대해 어느 것이 빠른지 예측하는 것이 불가능할 수 있다. TensorRT는 각 조합에 대한 tactic의 timing을 측정하고 가장 빠른 configuration을 찾아서 추론 중에 사용하도록 하는 것이 가능하다.

TensorRT에 custom tactics를 알려주려면 다음의 메소드를 구현해야 한다.
```c++
int32_t FooPlugin::getNbTactics() noexcept override
{
    return 2; // return 0 to disable custom tactics (default behavior)
}

int32_t FooPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
{
    tactics[0] = 1;
    tactics[1] = 2;
    return 0;
}
```

임의의 양수의 정수가 custom tactic value로 사용될 수 있다 (0은 TensorRT의 default tactic으로 예약되어 있음).

엔진을 빌드하는 중에 플러그인을 auto-tuning할 때, TensorRT는 `IPluginV3OneRuntime::setTactic`을 호출하여 이어지는 `enqueue()`에 사용할 tactic을 전달한다. 엔진이 deserialization되면 TensorRT는 플러그인이 생성된 후, `setTactic`을 한 번 호출하여 선택된 tactic을 플러그인에 전달한다. custom tactics를 사용하지 않더라도 `setTactic`은 default tactic value 0으로 호출된다.

# Sharing Custom Resources Among Plugins

TensorRT 10.0부터 key-value store는 문자열 키에 대해 사용자가 구현한 `IPluginResource` 객체를 저장하는 데 사용할 수 있는 plugin registry와 연결된다. 이를 사용하면 여러 플러그인 간의 상태나 일부 리소스를 공유할 수 있다.

예제를 통해 살펴보자.

## Example: Sharing Weights Downloaded Over a Network Among Different Plugins

여러 플러그인이 동일한 weight `W`에 액세스해야 된다고 가정해보자. 라이센스 제약으로 인해, 엔진이 실행될 때 이러한 weights를 다운로드하는 것이 선호될 수 있다. 그러나 `W`의 크기가 크므로 하나의 복사본만 다운로드하고 이 복사본을 액세스가 필요한 모든 플러그인 간에 공유하는 것이 바람직하다.

1. `IPluginResource`를 구현하는 `SharedWeights` 클래스를 구현한다.
2. Weights에 대한 액세스가 필요한 각 플러그인들은 `IPluginRegistry::acquirePluginResource(...)`를 호출하여 초기화된(다운로드된) `SharedWeights`의 인스턴스를 요청한다. <br>`IPluginResource* acquirePluginResource(char const* key, IPluginResource* resource)` <br>특정 key에 대해 `acquirePluginResource`가 처음 호출되면 TensorRT는 리소스로 전달된 객체 대신 제공된 plugin resource의 복사본을 등록한다. 즉, 등록된 객체는 `resource->clone()`을 호출하여 얻은 객체이다. 따라서, 복사본만 초기화하는 것이 가장 좋으며, 이 경우에 `IPluginResource::clone()`에서 weight download를 수행할 수 있다.
3. 각 플러그인은 weights 사용을 완료한 후, `IPluginRegistry::releasePluginResource()`를 호출하여 더 이상 weights를 사용하지 않는다고 신호를 보낼 수 있다. <br>`int32_t releasePluginResource(char const* key)` <br>TensorRT는 특정 key에 대한 `acquirePluginResource` 및 `releasePluginResource` 호출에 대한 reference counting을 수행하고 이 값이 0에 도달하면 `IPluginResource::release()`를 호출한다. 이 기능을 통해 모든 플러그인이 weight 사용을 마친 후, weights에 사용되는 메모리를 확보할 수 있다.

`SharedWeights` 구현 예시는 다음과 같다.
```c++
class SharedWeights : public IPluginResource
{
public:
    SharedWeights(bool init = false) {
        if (init) {
            cudaMalloc((void**)&cloned->mWeights, ...);
        }
    }

    int32_t release() noexcept override {
        try {
            if (mWeights != nullptr) {
                cudaFree(mWeights);
            }
        }
        catch {
            return -1;
        }
        return 0;
    }

    IPluginResource* clone() noexcept override {
        try {
            auto cloned = std::make_unique<SharedWeights>(true);
            //
            // Download the weights
            //
            // Copy to device memory
            cudaMemcpy(cloned->mWeights, ...);
        }
        catch {
            return nullptr;
        }
        return cloned.release();
    }

    ~SharedWeights() override {
        if (mWeights) {
            release();
        }
    }

    float* mWeights{nullptr};
};
```

`FooPlugin`이 weights에 액세스해야 된다고 가정한다. Weights는 추론 준비가 끝나면 요청할 수 있다. 이는 `IPluginV3OneRuntime::onShapeChange`에서 수행할 수 있고, 이는 build phase 및 runtime phase 모두에서 `enqueue()`가 될 플러그인에 대해 적어도 한 번은 호출된다.
```c++
int32_t onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
{
    SharedWeights w{};
    mW = static_cast<SharedWeights*>(getPluginRegistry()->acquirePluginResource("W", &w))->mWeights;
    return 0;
}
```

이렇게 얻은 weights (`mW`)는 이후에 이어지는 `enqueue()`에서 사용될 수 있다. 마무리를 위해서 plugin은 소멸자에서 release한다고 알려줄 수 있다. `IPluginV3`에는 `IPluginV2DynamicExt::terminate()`와 유사한 별도의 release resource 루틴이 없다.
```c++
FooPlugin::~FooPlugin() override
{
    try {
        getPluginRegistry()->releasePluginResources("W");
    }
    catch {
        // error handling
    }
}
```

위 코드는 weights 액세스가 필요한 모든 플러그인에 동일하게 사용될 수 있으며, reference counting 메커니즘을 통해 사용과 적절한 해제가 보장된다.

# Using Custom Layers When Importing a Model with a Parser

ONNX 파서는 자동으로 인식되지 않은 노드를 플러그인으로 임포트하려고 시도한다. 만약 플러그인 레지스트리에서 ONNX 노드와 `op_type`이 같은 플러그인이 발견되면 parser는 플러그인을 생성하기 위해서 해당 노드의 속성을 plugin creator에게 플러그인 field parameters로 전달한다. 기본적으로 parser는 플러그인 버전으로 "1"을 사용하고 네임스페이스는 ""을 사용한다. 이는 해당 ONNX 노드에서 `plugin_version`과 `plugin_namespace` 문자열 속성을 설정하여 이 동작을 오버라이드할 수 있다.

경우에 따라서 TensorRT로 임포트하기 전에 ONNX 그래프를 수정해야 할 필요가 있다. 예를 들어, 일련의 Ops를 플러그인 노드로 대체하는 경우에 이에 해당한다. 이를 위해서 [`ONNX GraphSurgeon`](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)을 제공한다. 사용 방법은 링크된 [예제](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph)에서 자세히 설명하고 있다.

더 많은 예제가 [onnx_packnet](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet) 샘플에서 제공된다.

# Plugin API Description

모든 새로운 플러그인은 `IPluginCreatorV3One`과 `IPluginV3` 클래스를 파생해야 한다. 또한, 새로운 플러그인은 plugin registry에 등록되어야 하며, `IPluginRegistry::registerCreator()`로 동적으로 등록되거나 `REGISTER_TENSORRT_PLUGIN(...)` 매크로로 정적으로 등록될 수 있다.

> **Note**: automotive safety user는 `REGISTER_TENSORRT_PLUGIN(...)` 대신 `REGISTER_SAFE_TENSORRT_PLUGIN(...)`을 사용해야 한다.

## `IPluginV3` API Description

아래는 `IPluginV3`의 함수들과 `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`의 함수들에 대한 정보이다.

`IPluginV3` 객체는 다른 capabilities로 구성되므로, `IPluginV3::getCapabilityInterface`는 생애주기 동안 언제든지 호출될 수 있다. Build phase에서 추가된 `IPluginV3` 객체는 모든 capability 타입(core, build, runtime)에 대해 유효한 capability interface를 리턴해야 한다. Runtime phase에 추가되는 객체는 build capability가 생략될 수 있다.

아래는 플러그인에 대한 정보를 식별하는데 사용되는 함수들이다. 플러그인의 생애주기 동안 모든 스테이지에서 호출될 수 있다.

- `IPluginV3OneCore::getPluginName` - Used to query for the plugin's name.
- `IPluginV3OneCore::getPLuginVersion` - Used to query for the plugin's version.
- `IPluginV3OneCore::getPluginNamespace` - Used to query for th eplugin's namespace.
- `IPluginV3OneBuild::getMetadataString` - Used to query for a string representation of any metadata associated with the plugin, such as the values of its attributes.

플러그인 레이어와 인접한 레이어를 연결하고 입출력 데이터 구조를 설정하기 위해 builder는 다음의 메소드들을 호출하여 output의 수와 이들의 shapes를 확인한다.

- `IPluginV3OneBuild::getNbOutputs` - Used to specify the number of output tensors.
- `IPluginV3OneBuild::getOutputShapes` - Used to specify the shapes of output as a function of the input shapes or constants. The exception is data-dependent shapes where an upper-bound and optimal tuning value is specified.
- `IPluginV3OneBuild::supportsFormatCombination` - Used to check if a plugin supports a given data type and format combination.
- `IPluginV3OneBuild::getOutputDataType` - Used to get the data types of the output tensors. The returned data types must have a format that is supported by the plugin.


플러그인 레이어는 다음의 data format들을 지원한다.
- `LINEAR` - FP32, FP16, BF16, FP8(E4M3), INT8, INT32 tensors
- `CHW32` - FP32, INT8 tensors
- `CHW2`, `HWC8`, `HWC16`, `DHWC8` - FP16 tensors
- `CHW4` - FP16, INT8 tensors
- `HWC8`, `HWC4`, `NDHWC8`, `NC2HW` - BF16 tensors
  
플러그인은 in-place로 계산하지 않고 입력 및 출력 텐서 외에 메모리 공간이 필요한 플러그인은 builder가 scratch space를 결정하고 사전에 할당하기 위해 호출하는 `getWorkspaceSize` 메소드를 사용하여 추가 메모리 요구 사항을 지정할 수 있다.

빌드 시, 최적의 configuration을 찾기 위해 레이어는 configured, executed, destroyed 된다. 최적의 configuration이 선택된 후, 선택된 tactic, 구체적인 shape/format 정보(data-dependent dimension 제외)가 플러그인에 전달되고,  이는 추론 어플리케이션의 생에주기 동안 필요한 만큼 여러 번 실행된다. 그리고 엔진이 제거될 때, 최종적으로 제거된다.

이러한 단계는 아래의 플러그인 메소드를 사용하여 builder와 runtime에 의해 제어된다. 추론 중 호출되는 메소드들은 (*)로 표시되어 있다. 다른 모든 메소드들은 builder에 의해서만 호출된다.

- `IPluginV3OneBuild::attachToContext`* - Used to request a plugin clone to be attached to an `ExecutionContext` and also to provide the opportunity for the plugin to access any context-specific resources.
- `IPluginV3OneBuild::getTimingCachedId` - Used to query for any timing cached ID that may be used by TensorRT. Enables timing caching if provided (disabled by default).
- `IPluginV3OneBuild::getValidTactics` - Used to query for any custom tactics the plugin may choose to use. The plugin will be profiled for each such tactic up to a maximum indicated by `IPluginV3OneBuild::getFormatCombinationLimit()`.
- `IPluginV3OneBuild::getFormatCombinationLimit` - Used to query for the maximum number of format combinations that my be timed for each tactic (for the default tactic `0` if no custom tactics are advertised).
- `IPluginV3OneBuild::configurePlugin` - Communicates the number of inputs and outputs, and their shapes, data types, and formats. The min, opt, and max of each input or output’s `DynamicPluginTensorDesc` correspond to the `kMIN`, `kOPT`, and `kMAX` value of the optimization profile that the plugin is being currently profiled for, with the `desc.dims` field corresponding to the dimensions of plugin inputs specified at network creation. Wildcard dimensions may exist during this phase in the `desc.dims` field. <br>At this point, the plugin may set up its internal state and select the most appropriate algorithm and data structures for the given configuration.
- `IPluginV3OneRuntime::setTactic`* - Communicates the tactic to be used during the subsequent `enqueue()`. If no custom tactics were advertised, this would always be `0`.
- `IPluginV3OneRuntime::onShapeChange`* - Communicates the number of inputs and outputs, and their shapes, data types and formats. The dimensions are concrete, except if data-dependent dimensions exist, which will be indicated by wildcards.
- `IPluginV3OneRuntime::enqueue`* - Encapsulates the actual algorithm and kernel calls of the plugin and provides pointers to input, output, and scratch space, and the CUDA stream to be used for kernel execution.
- `IPluginV3::clone` - This is called every time a new builder, network, or engine is created that includes this plugin layer. It must return a new plugin object with the correct parameters.

## `IPluginCreatorV3One` API Description

아래의 `IPluginCreatorV3One` 클래스 메소드들은 plugin registry로부터 적절할 플러그인을 찾거나 생성할 때 사용된다.

- `getPluginName` - This returns the plugin name and should match the return value of `IPluginV3OneCore::getPluginName`.
- `getPluginVersion` - Returns the plugin version. For all internal TensorRT plugins, this defaults to `1`.
- `getPluginNamespcae` - Returns the plugin namespace. Default can be "".
- `getFieldNamed` - To successfully create a plugin, it is necessary to know all the field parameters of the plugin. This method returns the `PluginFieldCollection` struct with the `PluginField` entries populated to reflect the field name and `PluginFieldType` (the data should point to `nullptr`).
- `createPlugin` - This method is used to create a plugin: it is passed a `PluginFieldCollection` and a `TensorRTPhase` argument.

Engine desrialization 중에 TensorRT는 `TensorRTPhase` 인자를 `TensorRTPhase::kRUNTIME`으로 설정하고 `PluginFieldCollection`을 `IPluginV3OneRuntime::getFieldsToSerialize()`에서 반환한 것과 동일한 `PluginField`로 채워서 이 메소드를 호출한다. 이때, TensorRT는 `createPlugin`이 반환한 플러그인 객체의 소유권을 갖는다.

또한, `createPlugin`을 호출하여 TensorRT 네트워크에 추가할 플러그인 객체를 생성할 수 있다. 이 경우, phase 인자를 `TensorRTPhase::kBUILD`로 설정하는 것이 좋다. `PluginFieldCollection`과 함께 전달된 데이터는 caller에 의해 할당되어야 하며, 프로그램이 삭제되기 전에 caller에 의해 해제되어야 한다. `createPlugin` 함수에 의해 반환된 플러그인의 객체의 소유권은 caller에게 있으며 반드시 caller가 제거해야 한다.

# Migrating V2 Plugins to IPluginV3

`IPluginV2`와 `IPluginV2Ext`는 TensorRT 8.5부터 지원이 중단되었고, `IPluginV2IOExt`, `IPluginV2DynamicExt`는 TensorRT 10.0부터 지원 중단되었다. 따라서, 새로운 플러그인은 `IPluginV3`로 구현되어야 하고, 이전 플러그인들은 리팩토링되어야 한다.

`IPluginV2DynamicExt` 플러그인을 `IPluginV3`로 마이그레이션할 때 아래 내용에 주의해야 한다.

- 플러그인과 연관된 plugin creator는 `IPluginCreatorV3One`으로 마이그레이션한다. 단순히 `IPluginCreator::deserializePlugin` 마이그레이션으로 구성되며, 이에 대한 자세한 내용은 [Plugin Serialization and Deserialization](#plugin-serialization-and-deserialization)에서 다루고 있다.
- `IPluginV2::iniialize()`, `IPluginV2::terminate()`, `IPluginV2::destroy()`와 동등한 메소드가 `IPluginV3`에는 없다. [Plugin Initialization and Termination](#plugin-initialization-and-termination)에서 이에 대한 내용을 다룬다.
- `IPluginV2Ext::detachFromContext()`와 등등한 메소드가 없다. [Accessing Context-Specific Resources Provided by TensorRT](#accessing-context-specific-resources-provided-by-tensorrt)에서 자세한 내용을 다룬다.
- `IPluginV3`에서 plugin serialization은 `IPluginV3OneRuntime::getFieldsToSerialize()`에 의해 TensorRT에 전달되는 `PluginFieldCollection`을 통해 이루어지며, deserialization은 TensorRT에 의해 `IPluginCreatorV3One::createPlugin(...)`에 다시 전달되는 동일한 `PluginFieldCollection`을 통해 이루어진다. 이에 대한 내용은 [Plugin Serialization and Deserialization](#plugin-serialization-and-deserialization)에서 자세히 다룬다.
- `IPluginV2DynamicExt`에서 `void`를 반환하는 메소드들은 `IPluginV3`에서는 status code를 반환한다 (e.g., `configurePlugin`).
- `supportsFormatCombination`과 `getWorkspaceSize`는 static descriptors(`PluginTensorDesc`) 대신 dynamic tensor descriptor(`DynamicPluginTensorDesc`)를 받는다.
- `IPluginV2DynamicExt::getOutputDimensions()`는 `IPluginV3OneBuild::getOutputShapes()`와 동일한 역할을 수행하며, 리턴 값이 아닌 output 매개 변수로 결과를 전달한다. 또한, per-output index querying가 아닌 one-shot querying으로 변경되었다. 비슷하게 `IPluginV2Ext::getOutputDataType`은 `IPluginV3OneBuild::getOutputDataTypes`가 동일한 역할을 한다.

## Plugin Initialization and Termination

`IPluginV2`는 initialization 및 termination을 위한`IPluginV2::initialize()`, `IPluginV2::terminate()`, `IPluginV2::destroy()` APIs를 제공했다. `IPluginV3`에서는 플러그인이 초기화된 상태로 구성되어야 한다. 사용했던 V2 플러그인에서 lazy initialization이 이루어지는 경우에는 `onShapeChange` 또는 `configurePlugin`에서 초기화를 하도록 지연할 수 있다. `IPluginV2::terminate()` 및 `IPluginV2::destroy()`에서 발생하는 모든 리소스 해제 또는 기타 종료 로직은 플러그인 클래스의 소멸자로 이동하면 된다. 파이썬에서는 예외적으로 C++의 소멸자의 역할을 하는 `IPluginV3.destroy()` API가 대안으로 제공된다.

## Accessing Context-Specific Resources Provided by TensorRT

`IPluginV2Ext::attachToContext()`를 통해 플러그인은 GPU allocator, cuDNN, cuBLAS 핸들과 같은 context-specific resources에 액세스할 수 있다. V3에서는 `IPluginV3OneRuntime::attachToContext()`가 유사한 기능을 제공하며 대신 `IPluginResourceContext`를 제공한다.

cuDNN 및 cuBLAS 핸들은 더 이상 `IPluginResourceContext`에서 제공되지 않는다. 이에 의존하는 플러그인은 자체 cuDN, cuBLAS 리소스를 초기화해야 한다. 플러그인 간에 cuDNN/cuBLAS 리소스를 공유하는 것이 선호되는 경우, IPluginResource에서 제공하는 기능과 plugin registry의 key-value store를 활용하여 이를 수행할 수 있다. 이에 대한 내용은 [Sharing Custom Resources Among Plugins](#sharing-custom-resources-among-plugins)에서 다룬다.

`IPluginV3OneRuntime::attachToContext(...)`는 clone-and-attach operation이다. 이는 runtime capability만이 아닌 전체 `IPluginV3` 객체를 클론하도록 요청받는다. 따라서 별도의 클래스로 구현된 경우, runtime capability 객체는 자신이 포함되는 `IPluginV3` 객체에 대한 참조를 보유해야 할 수도 있다.

`IPluginResourceContext`를 통해 얻은 모든 context-specific resource는 플러그인이 삭제될 때까지 사용될 수 있다. `IPluginV2Ext::detachFromContext()`에 구현된 모든 termination logic은 플러그인 소멸자로 이동해야 한다.

## Plugin Serialization and Deserialization

V2에서 serialization과 deserialization은 `IPluginV2::serialize`, `IPluginV2::getSerializationSize`, `IPluginCreator::deserializePlugin`의 구현으로 결정되었다. 3개의 메소드는 `IPluginV3OneRuntime::getFieldsToSerialize`와 `IPluginCreatorV3One::createPlugin`으로 대체된다. Raw buffer로부터의 writing/reading은 `PluginFieldCollection`을 구성하고 파싱하는 것으로 변환되었다.

`PluginFieldType`에 정의된 타입의 serialization은 TensorRT에 의해 처리된다. 커스텀 타입의 경우에는 `PluginFieldType::kUNKNOWN`으로 serialization될 수 있다.
```c++
struct DummyStruct {
    int32_t a;
    float b;
}

DummyPlugin() {
    // std::vector<nvinfer1::PluginField> mDataToSerialize;
    // int32_t mIntValue;
    // std::vector<float> mFloatVector;
    // DummyStruct mDummyStruct;
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("intScalar", &mIntvalue, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("floatVector", mFloatVector.data(), PluginFieldType::kFLOAT32, mFloatVector.size()));
    mDataToSerialize.emplace_back(PluginField("dummyStruct", &mDummyStruct, PluginFieldType::kUNKNOWN, sizeof(DummyStruct)));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

nvinfer1::PluginFieldCollection const* DummyPlugin::getFieldsToSerialize() noexcept override {
    return &mFCToSerialize;
}
```

## Migraing Older V2 Plugins to IPluginV3

`IPluginV2` 또는 `IPluginV2Ext`로부터 `IPluginV3`로 마이그레이션하는 경우, 먼저 `IPluginV2DynamicExt`로 마이그레이션한 다음 위의 단계에 따라 `IPluginV3`로 마이그레이션하는 것이 더 쉽다. `IPluginV2DynamicExt`의 새로운 기능은 다음과 같다.
```c++
virtual DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) = 0;

virtual bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) = 0;

virtual void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) = 0;

virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const = 0;

virtual int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) = 0;
```

- `getOutputDimension`은 주어진 inputs에 대한 output tensor dimensions의 표현을 구현한다.
- `supportsFormatCombination`은 플러그인이 지정된 I/O에 대한 format과 data type을 지원하는지 체크한다.
- `configurePlugin`은 `IPluginV2Ext`의 `configurePlugin`의 동작을 모방하지만 tensor descriptor를 받는다.
- `getWorkspaceSize`와 `enqueue`는 `IPluginV2Ext`와 동일하지만 tensor descriptor를 받는다.

# Coding Guidelines for Plugins

- **Memory Allocation**
  
  플러그인 내에서 할당된 메모리는 메모리 누수가 발생하지 않도록 반드시 해제되어야 한다. 플러그인의 생성자나 `onShapeChange`와 같은 이후 단계에서 리소스를 획득한 경우, 해당 리소스는 플러그인의 소멸자에서 해제되어야 한다. 또 다른 옵션으로는 `getWorkspaceSize`를 통해 필요한 추가 메모리 공간을 요청하는 것이다. 이 메모리는 `enqueue`에서 사용할 수 있다.

- **Add Checks to Ensure Proper Configurations and Validate Inputs**
  
  예상치 못한 플러그인 동작의 일반적인 원인은 적절하지 못한 configuration(e.g., invalid plugin attributes) 및 invalid inputs 이다. 따라서, 플러그인이 동작하지 않을 것으로 예상되는 경우, 초기 플러그인 개발 중에 checks 및 assertions를 추가하는 것이 좋다. 검사를 추가할 수 있는 위치는 다음과 같다.
  - `createPlugin` : plugin attrubtes checks
  - `configurePlugin/onShapeChange` : input dimension checks
  - `enqueue` : input valid checks

- **Return Null at Errors for Methods That Creates a New Plugin Object**
  `creaetPlugin`, `clone` 및 `attachToContext`와 같은 메소드는 새로운 플러그인 객체를 생성하고 반환할 것으로 예상할 수 있다. 이러한 메소드에서는 에러가 발생하거나 검사를 실패한 경우에 null 객체를 반환되는지 확인해야 한다. 그러면 플러그인이 잘못 구성되었을 때 null이 아닌 객체가 반환되지 않는다.

- **Avoid Device Memory Allocation in `clone()`**
  `clone`은 builder에서 여러 번 호출되므로 device memory allocation 비용이 상당히 클 수 있다. 한 가지 옵션은 persistent memory allocation을 수행하고, 플러그인을 사용할 준비가 되었을 때(예를 들어, `configurePlugin`에서) device에 복사하고, 플러그인이 파괴될 때 릴리즈하는 것이다.

- **Serializing Arbitrary Pieces of Data and Custom Types**
  
  `PluginField`의 `PluginFieldType::kUNKNOWN`을 사용하여 임의의 데이터 조각을 나타낼 수 있다. 이 경우, `PluginField`의 길이는 데이터가 가리키는 버퍼에 해당하는 바이트 수이어야 한다. Non-primitive types는 이러한 방식으로 serialization될 수 있다.

# Plugin Shared Libraries

TensorRT에는 어플리케이션에서 statically 로드할 수 있는 built-in 플러그인을 포함한다.

`REGISTER_TENSORRT_PLUGIN`과 `registerCreator` 인터페이스를 사용하여 TensorRT에 커스텀 플러그인을 명시적으로 등록할 수 있다. 그러나 TensorRT에서 플러그인 라이브러리의 등록을 관리하도록 하고 싶을 수 있고, 특히, plan 파일과 함께 플러그인 라이브러리를 직렬화하여 엔진이 생성될 때 자동으로 로드될 수 있다. 이느 특히 버전에 호환되는 엔진에서 플러그인을 포함하고 싶을 때 유용할 수 있고, 엔진을 빌드한 후 플러그인을 관리할 필요가 없게 된다. 이러한 이점을 얻으려면 TensorRT에 의해 인식하는 특정 entry point로 공유 라이브러리를 빌드할 수 있다.

## Generating Plugin Shared Libraries

플러그인에 대한 공유 라이브러리를 생성하려면, 라이브러리는 반드시 아래의 public symbols이 필요하다.
```c++
extern "C" void setLoggerFinder(ILoggerFinder& finder);
extern "C" IPluginCreator* const* getPluginCreators(int32_t& nbCreators) const;
```

`extern "C"`는 name mangling을 방지하는데만 사용되며 메소드 자체는 C++로 구현되어야 한다.

`setLoggerFinder()`는 라이브러리 내에서 플러그인 코드의 로긍일 위한 `ILoggerFinder`의 global pointer를 설정해야 한다. `getPluginCreators()`는 라이브러리에서 포함하는 plugin creators 리스트를 반환한다. 아래 파일이 이에 대한 예시를 보여준다.

- [vfcCommon.cpp](https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/vfcCommon.cpp)
- [vfcCommon.h](https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/vfcCommon.h)

플러그인 라이브러리를 엔진 플랜과 함께 직렬화하려면 `BuilderConfig`의 `setPluginToSerialize()`를 사용하여 TensorRT에 플러그인 라이브러리의 경로를 제공해야 한다.

버전 호환 엔진을 빌드할 때, 플랜 내에 플러그인을 패키징할 수도 있다. 패키징된 플러그인은 엔진과 동일한 lifetime을 가지며, 인젠 실행 시 자동으로 등록/해제된다.

## Using Plugin Shared Libraries

공유 라이브러리를 빌드할 때, builder가 엔진과 함께 라이브러리를 직렬화하도록 구성해주어야 한다. 그럼 다음에 엔진을 TensorRT에 로드하면 직렬화된 플러그인 라이브러리가 자동으로 로드되고 등록된다.

먼저, 엔진을 빌드하기 전에 builder와 함께 사용할 플러그인을 로드해야 한다.
```c++
for (size_t i = 0; i < nbPluginLibs; ++i)
{
    builder->getPluginRegistry().loadLibrary(pluginLibs[i]);
}
```

그런 다음, 플러그인이 엔진에 포함되거나 외부로 전달될 지를 결정한다. 플랜을 플러그인과 함께 직렬화하려면 다음과 같이 하면 된다.
```c++
IBuilderConfig* config = builder->createBuilderConfig();
...
config->setPluginsToSerialize(pluginLibs, nbPluginLibs);
```

또는, 엔진 외부에 플러그인을 유지할 수 있다. 이러한 라이브러리는 엔진이 배포될 때 엔진과 함께 이를 제공해야 하고 엔진을 역직렬화하기 전에 런타임에 명시적으로 로드해야 한다.
```c++
// In this example, getExternalPluginLibs() is a user-implemented method which retrieves the list of libraries to use with the engine 
std::vector<std::string> pluginLibs = getExternalPluginLibs();
for (auto const &pluginLib : pluginLibs)
{
    runtime->getPluginRegistry().loadLibrary(pluginLib.c_str())
}
```

# References

- [NVIDIA TensorRT Documentation: Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
- [GitHub: TensorRT Plugins](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins)
- [GitHub: TensorRT onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [Example: Replacing A Subgraph using onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph)
- [Example: onnx_packnet](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet)