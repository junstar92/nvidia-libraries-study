# Table of Contents

- [Table of Contents](#table-of-contents)
- [Extending TensorRT with Custome Layers](#extending-tensorrt-with-custome-layers)
- [Adding Custom Layers Using the C++ API](#adding-custom-layers-using-the-c-api)
  - [Example: Adding a Custom Layer with Dynamic Shape Support Using C++](#example-adding-a-custom-layer-with-dynamic-shape-support-using-c)
  - [Example: Adding a Custom Layer with INT8 I/O Support Using C++](#example-adding-a-custom-layer-with-int8-io-support-using-c)
- [Adding Custom Layers Using the Python API](#adding-custom-layers-using-the-python-api)
- [Using Custom Layers When Importing a Model with a Parser](#using-custom-layers-when-importing-a-model-with-a-parser)
- [Plugin API Description](#plugin-api-description)
  - [Migrating Plugins from TensorRT 6.x or 7.x to TensorRT 8.x.x](#migrating-plugins-from-tensorrt-6x-or-7x-to-tensorrt-8xx)
  - [`IPluginV2` API Description](#ipluginv2-api-description)
  - [`IPluginCreator` API Description](#iplugincreator-api-description)
- [Best Practices for Custom Layer Plugin](#best-practices-for-custom-layer-plugin)
  - [Coding Guidelines for Plugins](#coding-guidelines-for-plugins)
  - [Using Plugins in Implicit/Explicit Batch Networks](#using-plugins-in-implicitexplicit-batch-networks)
  - [Communicating Shape Tensors to Plugins](#communicating-shape-tensors-to-plugins)
- [Plugin Shared Libraries](#plugin-shared-libraries)
  - [Generating Plugin Shared Libraries](#generating-plugin-shared-libraries)
  - [Using Plugin Shared Libraries](#using-plugin-shared-libraries)
- [References](#references)

<br>

# Extending TensorRT with Custome Layers

TensorRT는 다양한 레이어를 지원하고 기능은 지속적으로 확장된다. 그러나 지원되는 레이어가 특정 요구 사항을 만족하지 못하는 경우가 있을 수 있다. 이런 경우, TensorRT는 **플러그인(plugin)** 이라는 커스텀 레이어를 구현하여 확장할 수 있다.

TensorRT에는 사용할 수 있는 몇 가지 플러그인이 이미 포함되어 있으며, 포함되어 있는 플러그인은 [GitHub: TensorRT plulgins](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins)에서 확인할 수 있다.

어플리케이션에서 TensorRT 플러그인을 사용하려면 `libinfer_plugin.so` (`nvinfer_plugin.dll` for windows) 라이브러리가 반드시 로드되어야 한다. 그리고 어플리케이션 코드에서 `initLibNvInferPlugin`을 호출하여 모든 플러그인을 등록해야 한다.

당연히 자신만의 플러그인(커스텀 레이어)를 구현하고 사용할 수도 있다.

<br>

# Adding Custom Layers Using the C++ API

커스텀 레이어는 TensorRT 플러그인의 베이스 클래스 중 하나를 상속받아 구현할 수 있다.

다른 타입/포맷의 I/O를 지원하거나 dynamic shapes을 지원하기 위해서 다양한 베이스 클래스가 있다. 아래 표는 제공되는 베이스 클래스 리스트를 보여준다.

||Introduced in TensorRT version?|Mixed I/O formats/types|Dynamic Shapes?|Supports implicit/explicit batch mode?|
|--|--|--|--|--|
|[`IPluginV2Ext`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html)|5.1 (deprecated since TensorRT 8.5)|Limited|No|Both|
|[`IPluginV2IOExt`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html)|6.0.1|General|No|Both|
|[`IPluginV2DynamicExt`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html)|6.0.1|General|Yes|Explicit batch mode only|

네트워크에서 구현한 플러그인을 사용하려면 먼저 TensorRT의 `PluginRegistry`에 먼저 등록을 해야 한다. 이떄, 플러그인을 직접 등록하는 것은 아니고 `PluginCreator`에서 파생된 플러그인의 팩토리 클래스의 인스턴스를 등록한다. 클러그인 creator에는 플러그인의 이름, 버전, 플러그인의 field 파라미터 정보들을 제공한다.

플러그인을 레지스트리에 등록하는 방법에는 두 가지 방법이 있다.

- TensorRT에서 제공하는 `REGISTER_TENSORRT_PLUGIN` 매크로를 사용하여 정적으로 레지스트리에 plugin creator를 등록한다. 이 매크로는 항상 플러그인을 default namespace ("")에 등록한다. (statically)
- `initLibNvInferPlugins`와 비슷한 방식으로 `registerCreator`를 호출하여 plugin registry에 동적으로 등록한다. 이 방법은 잠재적으로 더 적은 메모리 공간을 제공하며 고유한 네임스페이스에 등록될 수 있다. 이렇게 하면 다른 플러그인 라이브러리에서 name collisions가 발생하지 않는다. (dynamically)

`IPluginCreator::createPlugin()` 호출은 `IPluginV2` 타입의 플러그인 객체를 리턴한다. 이 플러그인은 TensorRT 네트워크의 `addPluginV2()`를 호출하여 네트워크에 추가할 수 있다. 아래는 이에 대한 예제 코드이다.
```c++
// Look up the plugin in the registry
auto creator = getPluginRegistry()->getPluginCreator(pluginName, pluginVersion);
const PluginFieldCollection* pluginFC = creator->getFieldNames();
// Populate the fields parameters for the plugin layer 
// PluginFieldCollection *pluginData = parseAndFillFields(pluginFC, layerFields); 
// Create the plugin object using the layerName and the plugin meta data
IPluginV2 *pluginObj = creator->createPlugin(layerName, pluginData);
// Add the plugin to the TensorRT network 
auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), pluginObj);
… (build rest of the network and serialize engine)
// Destroy the plugin object
pluginObj->destroy()
… (free allocated pluginData)
```

> 위 코드에서 `createPlugin()` 메소드는 힙에 새로운 플러그인 객체를 생성하고 해당 객체에 대한 포인터를 반환한다. 메모리 누수를 방지하려면 `pluginObj`를 삭제해주어야 한다.

직렬화(serialization) 중에 TensorRT 엔진은 `IPluginV2` 타입의 플러그인에 대한 정보(타입, 버전, 네임스페이스 등)을 내부에 저장한다. 역직렬화(deserialization)에서 TensorRT는 플러그인 레지스트리로부터 plugin creator를 조회하고, `IPluginCreator::deserializePlugin()`을 호출한다. 엔진이 destroy되면, 엔진 빌드 중에 생성된 플러그인 객체의 복사본은 `IPluginV2::destroy()` 메소드가 호출되어 엔진에 의해 destroy된다. 생성한 플러그인 객체가 네트워크에 추가된 이후 해제되었는지 확인하는 것은 사용자의 책임이다.

아래는 플러그인을 사용할 때 주의할 사항이다.

- Do not serialize all plugin parameters: only those required for the plugin to function correctly at runtime. Build time parameters can be omitted.
- Serialize and deserialize plugin parameters in the **same order**. During deserialization, verify that plugin parameters are either initialized to a default value or to the deserialized value. Uninitialized parameters result in undefined behavior.
- If you are an automotive safety user, you must call `getSafePluginRegistry()` instead of `getPluginRegistry()`. You must also use the `REGISTER_SAFE_TENSORRT_PLUGIN` macro instead of `REGISTER_TENSORRT_PLUGIN`.

## Example: Adding a Custom Layer with Dynamic Shape Support Using C++

Dynamic Shape를 지원하도록 `IPluginV2DynamicExt`를 상속받아 `BarPlugin` 구현을 간단히 살펴보자.

구현할 `BarPlugin`은 두 개의 입력과 2개의 출력을 가지는 플러그인이며, 첫 번째 출력은 두 번째 입력의 단순 복사본이고 두 번째 출력은 두 입력을 첫 번째 차원으로 concat한다. 모든 입출력의 데이터 타입과 포맷은 모두 동일하며, linear 포맷이다.

`BarPlugin`은 다음과 같이 `IPluginV2DynamicExt`를 상속받으며, `IPluginV2DynamicExt`의 가상 메소드들을 오버라이딩으로 구현한다.
```c++
class BarPlugin : public IPluginV2DynamicExt
{
    ... override virtual methods inherited from IPluginV2DynamicExt
};
```

아래 4개의 메소드는 Dynamic shapes에 영향을 받는 메소드들이다 ([`nvinfer1::IPluginV2DynamicExt`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html) 참조).

- `getOutputDimensions`
- `supportsFormatCombination`
- `configurePlugin`
- `enqueue`

`getOutputDimensions`에 대한 오버라이드에서는 입력 차원을 기준으로 출력 차원의 symbolic expression을 반환한다. `IExprBuilder`를 `getOutputDimensions`의 인자로 전달하여 입력에 대한 expressions로부터 출력에 대한 expressions를 빌드할 수 있다. 아래 예제 코드에서 두 번째 출력의 차원은 첫 번째 입력과 동일하므로 case 1에서는 새로운 expressions을 작성할 필요가 없다.
```c++
DimsExpr BarPlugin::getOutputDimensions(
    int outputIndex, 
    const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) override
{
    switch (outputIndex)
    {
    case 0: {
        // First dimension of output is sum of input first dimensions.
        DimsExpr output(inputs[0]);
        output.d[0] = 
            exprBuilder.operation(DimensionOperation::kSUM,
                inputs[0].d[0], inputs[1].d[0]);
        return output;
    }
    case 1:
        return inputs[0];
    default:
        throw std::invalid_argument("invalid output");
    }
}
```

`supportsFormatCombination`에 대한 오버라이드는 포맷의 조합이 허용되는지 여부를 나타내야 한다. 이 인터페이스에서는 입력과 출력을 "connections"로 균일하게 인덱싱한다. 첫 번째 입력은 인덱스 0부터 시작하여 순서대로 나머지 입력을 인덱싱하고, 마지막 입력 인덱스부터 이어서 출력을 인덱싱한다. 이 예제에서 두 입력은 각각 connection 0과 1이고 출력은 connection 2와 3이다.

TensorRT는 `supportsFormatCombination`을 사용하여 주어진 포맷과 타입의 조합이 적합한지를 체크한다. 이때 더 작은 인덱스의 connection에 대한 포맷과 타입이 적합한지 체크한다. 따라서, 오버라이드 구현에서는 더 작은 인덱스의 connections은 이미 조사되었다고 가정하고 `pos` 인덱스의 connection에 초점을 맞출 수 있다.
```c++
bool BarPlugin::supportsFormatCombination(
    int pos,
    const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override
{
    assert(0 <= pos && pos < 4);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    
    switch (pos)
    {
    case 0: return in[0].format == TensorFormat::kLINEAR;
    case 1: return in[1].type == in[0].type &&
                    in[1].format == TensorFormat::kLINEAR;
    case 2: return out[0].type == in[0].type &&
                    out[0].format == TensorFormat::kLINEAR;
    case 3: return out[1].type == in[0].type &&
                    out[1].format == TensorFormat::kLINEAR;
    }
    throw std::invalid_argument("invalid connection number");
}
```

> **Important**: 위 구현에서 `pos`보다 더 작은 인덱스의 connection에 대한 포맷과 타입을 검사해야 하며, 절대로 더 높은 인덱스의 connection에 대한 포맷과 타입을 검사해서는 안된다. 위 예제 코드에서는 connection 3을 체크하기 위해 case 3에서 connection 0을 사용하고 있으며, connection 0을 체크할 때에는 connection 3을 사용해서 비교하지 않는다.

TensorRT는 `configurePlugin`을 사용하여 런타임에 플러그인을 셋업한다. 만약 아무것도 할 필요가 없다면 다음과 같이 빈 상태로 두면 된다.
```c++
void BarPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs)
{
}
```

만약 플러그인에서 발생할 수 있는 최소 또는 최대 차원을 알아야 하는 경우에는 모든 입력과 출력에 대해 `DynamicPluginTensorDesc::min` 또는 `DyanmicPluginTensorDesc::max` 필드를 조사할 수 있다. 포맷과 build-time 차원 정보는 `DynamicPluginTensorDesc::desc`로 알아낼 수 있다. 모든 runtime dimensions는 -1로 나타내며, 실제 차원은 `BarPlugin::enqueue`에 제공된다.

마지막으로 `BarPlugin::enqueue`에 대한 오버라이드는 실제 작업을 수행한다. Dynamic shape이므로 `euqueue`에는 각 입력 및 출력의 실제 차원, 타입, 포맷을 설명하는 `PluginTensorDesc`가 전달된다.
```c++
int32_t enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    void const* const* inputs, void *const *outputs,
    void* workspace, cudaStream_t stream) override
{
    ...
}
```

## Example: Adding a Custom Layer with INT8 I/O Support Using C++

이번에는 custom-pooling 레이어에 대해 INT8 I/O로 확장하는 방법을 설명하는 `PoolPlugin` 구현 예제를 살펴본다.
```c++
class PoolPlugin : public IPluginV2IOExt
{
    ... override vitual methods inherited from IPluginV2IOExt.
};
```

대부분의 완전 가상 메소드들은 플러그인과 공통이며, INT8 I/O에 영향을 주는 메인 메소드들은 다음과 같다.

- `supportsFormatCombination`
- `configurePlugin`
- `enqueue`

`supportsFormatCombination`에 대한 오버라이드는 어떤 INT8 I/O가 허용되는지를 나타내며, 이전에 구현한 `BarPlugin`에서의 구현과 유사하다. 이 예제에서 지원되는 I/O 텐서 포맷은 FP32, FP16, INT8 데이터 타입의 linear CHW이며, I/O 텐서는 반드시 동일한 타입을 가진다.
```c++
bool PoolPlugin::supportsFormatCombination(
    int pos,
    const PluginTensorDesc* inOut,
    int nbInputs, int nbOUtputs) const override
{
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= ((inOut[pos].type == DataType::kFLOAT) ||
                  (inOut[pos].type == DataType::kHALF) ||
                  (inOut[pos].type == DataType::kINT8));
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}
```

> **Important:**
>
> - 만약 INT8 I/O 플러그인을 사용하는 네트워크에서 INT8 calibration이 반드시 사용된다면, TensorRT가 calibration을 수행할 때 FP32를 사용하므로 플러그인은 FP32 I/O를 지원해야 한다.
> - FP32 I/O variant가 지원되지 않거나 INT8 calibration이 사용되지 않는 경우, 필요한 모든 INT8 I/O 텐서의 스케일은 명시적으로 설정되어야 한다.
> Calibration은 플러그인 내부 텐서의 dynamic range를 결정할 수 없다. 양자화된 데이터에 대해 동작하는 플러그인은 반드시 내부 텐서에 대한 자체 dynamic range를 계산해야 한다.

TensorRT는 `configurePlugin`을 호출하여 `PluginTensorDesc`를 통해 플러그인에 정보를 전달한다. 전달하는 정보는 멤버 변수로 전달되고 직렬화/역직렬화된다.
```c++
void PoolPlugin::configurePlugin(
    const PluginTensorDesc* in, int nbInputs,
    const PluginTensorDesc* out, int nbOutputs)
{
    ...
    mPoolingParams.mC = mInputDims.d[0];
    mPoolingParams.mH = mInputDims.d[1];
    mPoolingParams.mW = mInputDims.d[2];
    mPoolingParams.mP = mOutputDims.d[1];
    mPoolingParams.mQ = mOutputDims.d[2];
    mInHostScale = in[0].scale >= 0.0F ? in[0].scale : -1.0F;
    mOutHostScale = out[0].scale >= 0.0F ? out[0].scale : -1.0F;
}
```

텐서 별 INT8 I/O 스케일은 `PluginTensorDesc::scale`로부터 얻을 수 있다.

`PoolPLugin::enqueue`는 마찬가지로 실제 작업을 수행한다.
```c++
int PoolPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    ...
    CHECK(cudnnPoolingForward(mCudnn, mPoolingDesc, &kONE, mSrcDescriptor, input, &kZERO, mDstDescriptor, output));
    ...
    return 0;
}
```

> prefix로 `m`이 붙은 변수들은 직접 선언하는 멤버 변수이다.

<br>

# Adding Custom Layers Using the Python API

파이썬 API에서의 방법은 [문서](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer_python)를 참조

<br>

# Using Custom Layers When Importing a Model with a Parser

ONNX 파서는 자동으로 인식되지 않은 노드를 플러그인으로 임포트하려고 시도한다. 만약 플러그인 레지스트리에서 ONNX 노드와 `op_type`이 같은 플러그인이 발견되면 parser는 플러그인을 생성하기 위해서 해당 노드의 속성을 plugin creator에게 플러그인 field parameters로 전달한다. 기본적으로 parser는 플러그인 버전으로 "1"을 사용하고 네임스페이스는 ""을 사용한다. 이는 해당 ONNX 노드에서 `plugin_version`과 `plugin_namespace` 문자열 속성을 설정하여 이 동작을 오버라이드할 수 있다.

경우에 따라서 TensorRT로 임포트하기 전에 ONNX 그래프를 수정해야 할 필요가 있다. 예를 들어, 일련의 Ops를 플러그인 노드로 대체하는 경우에 이에 해당한다. 이를 위해서 [`ONNX GraphSurgeon`](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)을 제공한다. 사용 방법은 링크된 [예제](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph)에서 자세히 설명하고 있다.

더 많은 예제가 [onnx_packnet](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet) 샘플에서 제공된다.

<br>

# Plugin API Description

모든 새로운 플러그인들은 `IPluginCreator`와 플러그인의 베이스 클래스 중 하나에서 클래스를 파생해야 한다. 또한, 새로운 플러그인은 `REGISTER_TENSORRT_PLUGIN(...)` 매크로를 호출하여 플러그인을 TensorRT Plugin Regsitry에 등록하거나 `initLibNvInferPlugins()`와 동일한 init function을 생성해야 한다.

## Migrating Plugins from TensorRT 6.x or 7.x to TensorRT 8.x.x

해당 포스팅에서 이에 대한 내용은 자세히 다루진 않는다. [문서](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#migrating-plugins-6x-7x-to-8x)의 내용을 참조 바람.

## `IPluginV2` API Description

아래의 내용들은 [`IPluginV2`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2.html) 클래스의 함수들에 대해 설명한다. 플러그인 레이어를 이웃 레이어에 연결하려면 input/output data structures를 셋업하고 builder가 아래의 플러그인 메소드를 호출하여 outputs의 수와 차원에 대해 검사한다.

- `getNbOutputs`

  출력 텐서의 갯수를 지정하는데 사용.

- `getOutputDimensions`
  
  입력 차원의 함수로 출력 차원을 지정하는데 사용.

- `supportsFormat`
  
  플러그인이 주어진 데이터 타입을 지원하는지 체크하는데 사용.

- `getOutputDataType`
  
  주어진 인덱스의 출력의 데이터 타입을 얻는데 사용. 반환되는 데이터 타입은 반드시 플러그인에서 지원하는 포맷을 가져야 하며, 플러그인에서 지원되는 데이터 포맷은 다음과 같다.
  - `LINEAR` single-precision(FP32), half-precision(FP16), integer(INT8), and integer(INT32) tensors
  - `CHW32` single-precision(FP32) and integer(INT8) tensors
  - `CHW2`, `HWC8`, `HWC16`, and `DHWC8` half-precision(FP16) tensors
  - `CHW4` half-precision(FP16) and integer(INT8) tensors
  
플러그인은 in-place로 계산하지 않고 입력 및 출력 텐서 외에 메모리 공간이 필요한 플러그인은 builder가 scratch space를 결정하고 사전에 할당하기 위해 호출하는 `getWorkspaceSize` 메소드를 사용하여 추가 메모리 요구 사항을 지정할 수 있다.

빌드 및 추론 시간 동안, 플러그인 레이어가 구성되고 실행되며, 당연히 여러 번 실행될 수도 있다. 빌드 시, 최적의 구성을 찾기 위해 레이어가 configured, initialized, executed, terminated 된다. 플러그인에 대한 최적의 포맷을 선택한 후, 플러그인을 다시 한번 configured하고 한 번 초기화한 뒤에 추론을 필요한 만큼 수행한다. 그리고 엔진이 destroy되면 최종적으로 terminated된다. 이와 같은 단계는 아래의 메소드들을 통해 builder와 engine에 의해 제어된다.

- `configurePlugin`
  
  입출력의 수, 모든 입력과 출력의 차원 및 데이터 타입, 브로드캐스트 정보, 선택된 플러그인의 포맷, 최대 배치 사이즈를 전달한다. 이 시점에서 플러그인은 내부 상태를 셋업하고 주어진 configuration에서 가장 적절한 알고리즘과 데이터 타입을 선택한다. 이 메소드에서의 **resource allocation** 은 메모리 누수를 일으키므로 허용되지 않는다.

- `initialize`
  
  이 시점에서 configuration을 알려져 있고 inference engine이 생성되는 중이다. 따라서 플러그인은 내부 데이터 구조를 셋업할 수 있고 실행을 준비할 수 있다.

- `enqueue`
  
  실제 알고리즘 및 커널 호출을 캡슐화하고 커널 실행에서 사용되는 runtime batch size, 입출력에 대한 pointers, scatch space, CUDA 스트림을 제공한다.

- `terminate`
  
  Engine context가 destroy되고, 플러그인이 가지고 있는 모든 리소스가 해제되어야 한다.

- `clone`
  
  이 메소드는 새로운 builder, network, 또는 engine이 생성될 때마다 호출된다. 올바른 파라미터가 있는 새로운 플러그인 객체를 반환해야 한다.

- `destroy`
  
  새로운 플러그인 객체가 할당될 때마다 할당되었던 플러그인 객체 및 다른 메모리를 destroy하는데 사용된다. 이 메소드는 builder, network, 또는 engine이 destroy될 때마다 호출된다.

- `set/getPluginNamespace`
  
  이 메소드는 플러그인 객체가 속해있는 라이브러리 네임스페이스를 설정하는데 사용된다. 같은 클러그인 라이브러리의 모든 플러그인 객체는 같은 네임스페이스를 갖는다.

[`IPluginV2Ext`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html)는 브로드캐스트 입력 및 출력을 처리할 수 있는 플러그인을 지원한다. 이 기능을 위해 아래의 메소드들이 구현되어야 한다.

- `canBroadcastInputAcrossBatch`
  
  이 메소드는 각 배치 전체에서 의미론적으로 브로드캐스트되는 각 입력 텐서에 대해 호출된다. 만약 `canBroadcastInputAcrossBatch`가 `true`를 반환한다면 TensorRT는 입력 텐서를 복제하지 않는다. 플러그인에는 배치 전체에서 공유해야 하는 single copy가 있다. 만약 `false`를 반환한다면 TensorRT는 nonbroadcasted tensor처럼 보이도록 하기 위해서 입력 텐서를 복제한다.

- `isOutputBroadcastAcrossBatch`

  이 메소드는 각 출력 인덱스에 대해 호출된다. (The plugin should return true the output at the given index and is broadcast across the batch.)

- `IPluginV2IOExt`

  이 메소드는 `initialize()`를 호출하기 전에 builder에 의해 호출된다. 이는 레이어가 I/O PluginTensorDesc 및 최대 배치 크기를 기반으로 알고리즘을 선택할 수 있는 기회를 제공한다.

> `IPluginV2`를 베이스로 하는 플러그인은 engine level에서 공유되며 execution context level에서는 공유되지 않는다. 그러므로 여러 스레드에서 동시에 사용할 수 있는 플러그인은 스레드로부터 안전한 방식으로 리소스를 관리해야 한다. `IPluginV2Ext`를 베이스로 하는 플러그인 및 파생 인터페이스는 `ExecutionContext`가 생성될 때 복제되므로 필요하지 않다.

## `IPluginCreator` API Description

다음은 [`IPluginCreator`](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#iplugincreator) 클래스의 메소드들이며 플러그인 레지스트리로부터 적절한 플러그인을 찾거나 생성하는데 사용된다.

- `getPluginName`
  
  플러그인 이름을 반환하며 `IPluginExt::getPluginType`의 반환값과 일치해야 한다.

- `getPluginVersion`
  
  플러그인 버전을 반환한다.

- `getFieldName`
  
  성공적으로 플러그인을 생성하기 위해서 플러그인의 모든 field parameters를 알아야 한다. 이 메소드는 `PluginFieldCollection` 구조체를 반환하며, 이 구조체에는 field name과 `PluginFieldType`(data 멤버는 `nullptr`를 가리킴)를 갖는 `PluginField` 항목이 있다.

- `createPlugin`

  이 메소드는 `PluginFieldCollection` 인자를 사용하여 플러그인을 생성하는데 사용된다. `PluginField`의 data field는 실제 데이터를 가리킨다.

  **Note:** `createPlugin` 함수로 전달되는 데이터는 caller에 의해 할당되어야 하며 프로그램이 종료될 때 caller에 의해서 해제되어야 한다. `createPlugin`으로부터 반환되는 플러그인 객체의 오너쉽은 caller에게 전달되며 마찬가지로 해제되어야 한다.

- `deserializePlugin`
  
  이 메소드는 플러그인 이름과 네임스페이스에 기반하여 TensorRT 엔진에 의해서 내부적으로 호출된다. 추론에 사용되는 플러그인 객체가 리턴되며 이 함수에 의해 생성된 플러그인 객체는 TensorRT 엔진이 파괴될 때, 엔진에 의해서 파괴된다.

- `set/getPluginNamespace`
  
  이 메소드는 creator instance가 속한 네임스페이스를 설정하는데 사용된다.

<br>

# Best Practices for Custom Layer Plugin

## Coding Guidelines for Plugins

- **Memory Allocation**
  
  플러그인 내에서 할당된 메모리는 메모리 누수가 발생하지 않도록 반드시 해제되어야 한다. 만약 `initialize()` 함수에서 리소스를 얻었다면, 이 리소스는 `terminate()` 함수에서 해제되어야 한다. 다른 모든 할당은 플러그인 클래스의 destructor 또는 `destroy()` 메소드에서 적절하게 해제되어야 한다.

- **Add Checks to Ensure Proper Configurations and Validate Inputs**
  
  예상치 못한 플러그인 동작의 일반적인 원인은 적절하지 못한 configuration(e.g., invalid plugin attributes) 및 invalid inputs 이다. 따라서, 플러그인이 동작하지 않을 것으로 예상되는 경우, 초기 플러그인 개발 중에 checks 및 assertions를 추가하는 것이 좋다. 검사를 추가할 수 있는 위치는 다음과 같다.
  - `createPlugin` : plugin attrubtes checks
  - `configurePlugin` : input dimension checks
  - `enqueue` : input valid checks

- **Return Null at Errors for Methods That Creates a New Plugin Object**
  `createPlugin`, `clone`, `desrializePlugin`은 새로운 플러그인 객체를 생성하거나 반환한다. 이 메소드들에서 모든 에러의 경우나 실패인 경우 null 객체가 반환되도록 해야 한다.


- **Avoid Device Memory Allocation in `clone()`**
  `clone` 메소드는 builder에 의해서 여러 번 호출되므로 이곳에서의 device memory 할당 비용은 상당하다. Persistant memory allocation은 `initialize`에서 발생되는 것이 좋으며, 플러그인이 사용할 준비가 되면 device로 복사하고, `terminate`에서 할당을 해제하는 것이 좋다.

## Using Plugins in Implicit/Explicit Batch Networks

TensorRT 네트워크는 implicit batch mode 또는 explicit batch mode로 생성할 수 있다. implicit/explicit batch mode 네트워크에서 플러그인 동작과 관련한 다음 내용들을 기억하면 유용하다.

- `IPluginV2DynamicExt`를 구현한 플러그인은 오직 explicit batch mode의 네트워크에만 추가될 수 있다.
- Non-`IpluginV2DynamicExt` 플러그인은 두 모드의 네트워크에 추가될 수 있다.

**Important:** 비록 non-`IPluginV2DynamicExt` 플러그인이 explicit batch mode 네트워크와 호환된다고 하더라도 그 구현은 네트워크의 타입과 독립적이어야 한다. 따라서, explicit batch mode 네트워크에서 이러한 플러그인을 사용하는 경우, 첫 번째 입력의 첫 번째 차원은 플러그인에 전달되기 전에 batch dimension으로 유추된다. TensorRT는 플러그인에 입력으로 전달되기 전에 방금 식별한 첫 번째 차원을 pop하고 플러그인의 모든 출력의 앞에 push한다. 즉, `getOutputDimensions`에서 batch dimension을 지정하면 안된다.

## Communicating Shape Tensors to Plugins

TensorRT plugin API는 플러그인에 대한 shape 텐서의 direct input이나 direct output을 지원하지 않는다. 그러나 이러한 제한은 empty tensors를 사용하여 해결할 수 있다. 관심있는 차원과 zero dimension을 가진 dummy input tensor를 사용하면 입력이 실제로 공간을 차지하지 않게 된다.

예를 들어, 플러그인의 output shape를 계산하기 위해 2-element 1D shape tensor 값 [P, Q]를 알아야 한다고 가정해보자. Shape tensor [P, Q]를 전달하는 대신 차원이 [0, P, Q]인 execution tensor를 더미 입력으로 갖도록 플러그인을 설계한다. TensorRT는 플러그인에게 더미 입력의 차원을 알려준다. Empty tensor이므로 아주 작은 공간만 차지하는 주소가 사용된다.

네트워크에서 zero-stride slice를 사용하거나 또는 empty tensor를 reshaping하여 dummpy input tensor를 생성한다. 아래 코드는 zero-stride slice를 사용하는 메커니즘을 보여준다.
```c++
// Shape tensor of interest. Assume it has the value [P, Q].
ITensor* pq = ...;

// Create an empty-tensor constant with dimensions [0, 1, 1].
// Since it's empty, the type doesn't matter, but let's assum float.
ITensor* c011 = network->addConstant({3, {0, 1, 1}}, {DataType::kFLOAT, nullptr, 0})->getOutput(0);

// Create shape tensor that has the value [0, P, Q]
static int32_t const intZero = 0;
ITensor* z = network->addConstant({1, {1}}, {DataType::kINT32, &intZero, 1})->getOutput(0);
ITensor* concatInputs[] = {z, pq};
IConcatenationLayer* zpq = network->addConcatenation(concatInputs, 2);
zpq->setAxis(0);

// Create zero-stride slice with output size [0, P, Q]
Dims z3{3, {0, 0, 0}};
ISliceLayer* slice = network->addSlice(*c011, z3, z3, z3);
slice->setInput(2, *zpq->getOutput(0));
```

여기서 `slice->getOutput(0)`은 플러그인의 더미 입력으로 사용한다.

만약 `ISuffleLayer`를 사용하여 empty tensor를 생성한다면, reshape 차원에서 0에 대한 특별한 해석을 해제해주어야 한다. 즉, `setZeroIsPlaceHolder(false)`를 호출해야 한다.

<br>

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

<br>

# References

- [NVIDIA TensorRT Documentation: Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
- [GitHub: TensorRT Plugins](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins)
- [GitHub: TensorRT onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [Example: Replacing A Subgraph using onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph)
- [Example: onnx_packnet](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet)