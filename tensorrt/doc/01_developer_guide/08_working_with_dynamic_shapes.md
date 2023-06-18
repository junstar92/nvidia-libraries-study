# Table of Contents

- [Table of Contents](#table-of-contents)
- [Working with Dynamic Shapes](#working-with-dynamic-shapes)
- [Specifying Runtime Dimensions](#specifying-runtime-dimensions)
- [Named Dimensions](#named-dimensions)
- [Dimension Constraint using IAssertionLayer](#dimension-constraint-using-iassertionlayer)
- [Optimization Profiles](#optimization-profiles)
- [Dynamically Shaped Output](#dynamically-shaped-output)
  - [Looking up Binding Indices for Multiple Optimization Profiles](#looking-up-binding-indices-for-multiple-optimization-profiles)
  - [Binding For Multiple Optimization Profiles](#binding-for-multiple-optimization-profiles)
- [Layer Extensions For Dynamic Shapes](#layer-extensions-for-dynamic-shapes)
- [Restrictions For Dynamic Shapes](#restrictions-for-dynamic-shapes)
- [Execution Tensors Versus Shape Tensors](#execution-tensors-versus-shape-tensors)
  - [Formal Inference Rules](#formal-inference-rules)
- [Shape Tensor I/O (Advanced)](#shape-tensor-io-advanced)
- [INT8 Calibration with Dynamic Shapes](#int8-calibration-with-dynamic-shapes)
- [References](#references)

<br>

# Working with Dynamic Shapes

_Dynamic Shapes_ 는 일부 또는 전체 차원을 런타임에 지정하는 기능이다. C++과 Python 인터페이스에서 모두 지원한다.

dynamic shapes를 사용하도록 엔진을 빌드하는 방법을 간략하게 살펴보자.

1. 먼저 netowk definition은 implicit batch dimensions를 가지도록 하면 안되며, 다음과 같이 explicit batch로 `INetworkDefinition`을 생성한다.
```c++
IBuilder::createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
```

2. 입력 텐서에서 런타임에 결정할 차원의 값을 placeholder로 `-1`로 지정한다.
3. 하나 또는 하나 이상의 _optimization profiles_ 를 지정한다. Optimization profile은 런타입에 결정되는 차원을 갖는 입력에서 허용되는 차원의 범위와 auto-tuner가 최적화할 차원을 지정한다. 이에 대한 내용은 [Optimization Profiles](#optimization-profiles)에서 자세하게 살펴본다.
4. 엔진을 빌드하고, 생성된 빌드를 통해 추론을 수행한다.
    1. 엔진으로부터 execution context를 생성한다 (dynamic shape를 사용하지 않을 때와 동일)
    2. 3번 과정에서 지정한 optimization profiles 중 하나를 지정한다.
    3. execution context에서 입력 차원 크기를 지정한다. 입력 차원이 설정되면 주어진 입력 차원에 대해 TensorRT가 계산한 출력 차원 크기를 얻을 수 있다.
    4. Enqueue work

만약 입력 차원 크기를 바꾸고 싶다면, 4-2와 4-3을 반복하면 되고, 입력 차원이 변경되지 않는다면 이를 반복할 필요가 없고 4-4만 반복하면 된다.

Preview feature (`PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805`)가 활성화될 때, 잠재적으로 dynamically shaped network에는 다음의 효과가 있다.

- reduce the engine build time.
- reduce runtime, and
- decrease device memory usage and engine size.

> `kFASTER_DYNAMIC_SHAPES_0805`를 활성화하여 혜택을 볼 가능성이 가장 높은 모델은 Transformer 기반 모델 및 dynamic control flows를 포함하는 모델이라고 문서에서 언급하고 있다.

<br>

# Specifying Runtime Dimensions

네트워크를 빌드할 때, `-1`의 값을 사용하여 입력 텐서에 대한 runtime dimension을 나타낸다. 예를 들어, `foo`라는 이름의 3D 입력 텐서를 생성하고, 마지막 두 차원이 런타임에 지정되고 첫 번째 차원만 빌드 시 고정되도록 하려면 다음과 같이 실행한다.
```c++
network->addInput("foo", DataType::kFLOAT, Dims3(3, -1, -1));
```

런타임에서는 반드시 optimization profile을 선택한 뒤, 입력 차원을 설정해주어야 한다. 아래 예제 코드는 입력 차원을 [3, 150, 250]으로 설정한다. 아래 코드는 optimization profile을 선택했다고 가정한다.
```c++
context->setInputShape("foo", Dims{3, {3, 150, 250}});
```
> Binding index를 통해 input shape를 지정하는 `setBindingDimensions()` 메소드도 있는데, `setInputShape()`가 도입되면서 TensorRT 8.5에서 deprecated 되었다.

런타임에서 binding dimensions에 대해 엔진을 쿼리하면 네트워크를 빌드할 때 사용된 것과 동일한 차원이 반환된다. 즉, 각 runtime dimension의 값은 `-1`이다. 예를 들어, 아래 호출은
```c++
engine->getTensorShape("foo");
```
차원이 `{3, -1, -1}`인 `Dims`를 리턴한다.

각 execution context에서 지정된 실제 차원을 얻으려면, execution context를 쿼리하면 된다.
```c++
context->getTensorShape("foo");
```
위 코드는 차원이 `{3, 150, 250}`인 `Dims`를 리턴한다.

입력에 대한 `setInputShape`의 리턴값은 해당 입력에 대해 설정된 optimization profile에 대한 일관성만 나타낸다. 모든 input binding dimensions가 지정된 후, 네트워크의 output bindings의 차원을 쿼리하여 전체 네트워크가 dynamic input shapes와 관련하여 일관성이 있는지 확인할 수 있다. 아래 코드는 `bar`라는 이름의 출력의 차원을 쿼리하는 예이다.
```c++
Dims outDims = context->getTensorShape("bar");

if (outDims.nbDims == -1) {
    gLogError << "Invalid network output, this might be caused by inconsistent input shape." << std::endl;
    // abort inference
}
```

만약 k번째 차원이 data-dependent라면, 예를 들어, `INonZeroLayer`의 output에 따라 다르다면, `outDims.d[k]`는 -1이다. 이에 대한 자세한 내용은 [Dynamically Shaped Output](#dynamically-shaped-output)에서 이러한 output을 다루는 방법에 대해서 확인할 수 있다.

<br>

# Named Dimensions

Constant와 runtime dimesions는 모두 이름을 가진다. 이로 인해 다음의 이점이 있다.

- 차원의 이름을 사용하여 에러 메세지를 출력한다. 예를 들어, 입력 텐서 `foo`가 `[n, 10, m]` 차원을 가질 때, `(#2 (SHAPE foo))`라는 출력보다 `m`이라는 이름으로 출력하면 더욱 쉽게 알아볼 수 있다.
- 같은 이름의 차원은 묵시적으로 동일하다. 이를 통해 optimizer가 더 효율적인 엔진을 생성하고, 런타임에서 미스매칭된 차원을 진단할 수 있다. 예를 들어, 두 입력 차원이 각각 `[n, 10, m]`과 `[n, 13]`일 때, optimizer는 두 입력의 lead dimensions이 항상 같다는 것을 알고 있다. 그리고 우연히 서로 다른 `n`을 사용하는 엔진이 있다면 에러를 리포트한다.

런타임에서 항상 동일하다면, contant와 runtime dimensions에 동일한 이름을 사용할 수 있다.

아래 예제 코드는 텐서의 3번째 차원 이름을 `m`으로 지정한다.
```c++
tensor->setDimensionName(2, "m");
```
그리고, 아래 메소드로 차원의 이름을 쿼리할 수 있다.
```c++
tensor->getDimensionName(2);
// return the name of the third dimension of tensor,
// or nulltpr if it does not have a name.
```

> ONNX 파일을 통해 input network를 임포트하면, ONNX parser가 차원의 이름을 ONNX file에서의 이름으로 자동으로 설정한다. 따라서, 두 개의 dynamic dimensions가 런타임에서 동일하다면 ONNX file을 export할 때, 이러한 차원 이름을 동일하게 지정하는 것이 좋다.

<br>

# Dimension Constraint using IAssertionLayer

경우에 따라 두 개의 dynamic dimensions가 동일하지 않지만, 런타임에 동일하다고 보장된다. TensorRT에게 두 차원이 동일하다는 것을 알려주면 더 효율적인 엔진을 구축하는데 도움이 될 수 있다. TensorRT에서 이러한 제약을 전달하는 방법에는 두 가지가 있다.

- [Named Dimensions](#named-dimensions)에서 언급한 것과 같이 차원의 이름을 동일하게 지정한다.
- `IAssertionLayer`를 사용하여 해당 제약을 표현한다. 이 기법은 더 까다로운 제약을 전달할 수 있어서 더 일반적이라고 볼 수 있다.

예를 들어, 텐서 A의 첫 번째 차원이 텐서 B의 첫 번째 차원보다 1 이상 더 크다고 보장하려면, 다음과 같이 제약을 구성하면 된다.
```c++
// Assume A and B are ITensor* and n is a INetworkDefinition&
auto shapeA = n.addShape(*A)->getOutput(0);
auto firstDimOfA = n.addSlice(*shapeA, Dims{1, {0}}, Dims{1, {1}}, Dims{1, {1}})->getOutput(0);

auto shapeB = n.addShape(*B)->getOutput(0);
auto firstDimOfB = n.addSlice(*shapeB, Dims{1, {0}}, Dims{1, {1}}, Dims{1, {1}})->getOutput(0);

static int32_t const oneStorage{1};
auto one = n.addConstant(Dims{1, {1}}, Weights{DataType::kINT32, &oneStorage, 1})->getOutput(0);

auto firstDimsOfBPlus1 = n.addElementWise(*firstDimOfB, *one, ElementWiseOperation::kSUM)->getOutput(0);
auto areEqual = n.addElementWise(*firstDimOfA, *firstDimOfBPlus1, ElementWiseOperation::kEQUAL)->getOutput(0);

n.addAssertion(*areEqual, "oops");
```

만약 런타임에서 해당 assertion을 위반하면 TensorRT는 에러를 던진다.

<br>

# Optimization Profiles

_Optimization profile_ 은 각 네트워크 입력의 차원 범위와 auto-tuner가 최적화하는데 사용할 차원을 지정한다. Runtime dimensions를 사용할 때, 빌드 시 적어도 하나에 optimization profile을 생성해야 한다. 둘 이상의 profiles를 지정할 때, 두 profiles은 서로 범위가 구분되거나 겹칠 수 있다.

예를 들어, 한 profile의 차원 범위는 minimum이 `[3, 100, 200]`, maximum이 `[3, 200, 300]`, 그리고 optimization 차원이 `[3, 150, 250]`가 되도록 지정하고, 다른 profile은 min, max, optimization 차원을 각각 `[3, 200, 100]`, `[3, 300, 400]`, `[3, 250, 250]`으로 지정할 수 있다.

`min`, `max`, `opt`로 지정되는 차원에 기반하여 다른 profiles의 메모리 사용량이 극적으로 변할 수 있다. `MIN=OPT=MAX`에서만 동작하는 tactics를 갖는 몇몇 연산들이 있으며, 이러한 연산에서 `MIN`,`OPT`,`MAX`의 값이 다르다면 이러한 tactic은 비활성화된다.

Optimization profile을 만드려면 먼저, `IOptimizationProfile`을 구성한다. 그런 다음, min, opt, max 차원을 설정하고, profile을 network configuration에 추가한다. Optimization profile에 의해 정의되는 shapes는 반드시 네트워크에 유효한 input shapes를 정의해야 한다. 아래 예제 코드는 `foo`라는 이름의 입력 텐서의 첫 번째 profile을 생성하고 이를 cnofiguration으로 추가하는 방법을 보여준다.
```c++
IOptimizationProfile* profile = builder->createOptimizationProfile();
profile->setDimensions("foo", OptProfileSelector::kMIN, Dims3(3, 100, 200));
profile->setDimensions("foo", OptProfileSelector::kOPT, Dims3(3, 150, 250));
profile->setDimensions("foo", OptProfileSelector::kMAX, Dims3(3, 200, 300));

config->addOptimizationProfile(profile);
```

런타임에서 입력 차원을 설정하기 전에 반드시 하나의 optimization profile을 설정해야 한다. Profiles는 추가된 순서로 넘버링되어 있으며 `0`부터 시작한다. 각 execution context는 별도의 optimization profile을 사용해야 한다.

아래 예제 코드는 첫 번째 optimization profile을 선택하고 사용한다.
```c++
context->setOptimizationProfileAsync(0, stream);
```
위 코드에서 `stream`은 이후 `enqueue()`, `enqueueV2()`, `enqueueV3()` 호출에서 동일하게 사용된다.

관련된 CUDA 엔진에 dynamic inputs이 있는 경우, 소멸되지 않은 다른 execution context에서 사용되고 있지 않는 optimization profile을 한 번 이상 설정해야 한다. 엔진에 대해 생성된 첫 번째 execution context의 경우에는 profile 0이 암묵적으로 선택된다.

`setOptimizationProfileAsync()`를 호출하여 profiles 간에 전환할 수 있다. 이 호출은 `enqueue()`, `enqueueV2()`, `enqueueV3()` 작업이 현재 context에서 완료된 후 호출해야 한다. 여러 개의 execution context가 동시에 실행되는 경우, 다른 execution context에서 해제된(사용되지 않는) profile로 전환할 수 있다.

`setOptimizationProfile()` API는 deprecated 되었다. 이를 사용하여 optimization profile을 전환하면 이어지는 `enqueue()`, `enqueueV2()` 작업에서 GPU 메모리 복사가 발생할 수 있다. 이러한 호출을 방지하려면 `setOptimizationProfileAsync()` API를 사용해야 한다.

<br>

# Dynamically Shaped Output

만약 네트워크의 출력이 dynamic shape라면 output memory를 할당하는 여러 방법들이 있다.

만약 output의 차원이 input의 차원으로부터 계산이 가능하다면, `IExecutionContext::getTensorShape()`를 사용하여 input 텐서와 [Shape Tensos I/O](#shape-tensor-io-advanced)의 차원을 제공하여 output의 차원을 얻는다. 만약 필요한 정보를 제공하는 것을 잊었다면, `IExecutionContext::inferShapes()` 메소드를 사용하면 된다.

위의 경우가 아니라면, 즉, output의 차원을 미리 계산할 수 있는지 또는 `enqueueV3`를 호출하는 지 알 수 없는 경우에는 `IOutputAllocator`를 output에 연결시켜준다. 구체적인 방법은 다음과 같다.

1. `IOutputAllocator`를 상속받는 파생 클래스를 생선한다.
2. `reallocateOutput`과 `notifyShape` 메소드를 오버라이딩한다. TensorRT는 output memory를 할당할 필요가 있을 때에는 `reallocateOutput`을 호출하고, output의 크기를 알고 있을 때에는 `notifyShape`를 호출한다. 예를 들어, `INonZeroLayer` 레이어의 output에 대한 메모리는 해당 레이어가 실행되기 전에 할당된다.

`IOutputAllocator`를 상속받는 파생 클래스의 구체적인 예시는 다음과 같다. `reallocateOutput`을 구현하는 방법들에 대해서는 아래에서 다시 설명한다.
```c++
class MyOutputAllocator : nvinfer1::IOutputAllocator
{
public:
    void* reallocateOutput(
        char const* tensorName, void* currentMemory, 
        uint64_t size, uint64_t alignment) override
    {
        // Allocate the output. Remember it for later use.
        outputPtr = ... depends on strategy, as discussed later...
        return outputPtr;
    }

    void notifyShape(char const* tensorName, Dims const& dims)
    {
        // Remember output dimensions for later use.
        outputDims = dims;
    }

    // Saved dimensions of the output
    Dims outputDims{};

    // nullptr if memory could not be allocated
    void* outputPtr{nullptr};
};
```

이렇게 구현한 클래스는 다음과 같이 사용할 수 있다.
```c++
std::unordered_map<std::string, MyOutputAllocator> allocatorMap;

for (const char* name : names of outputs)
{
    Dims extent = context->getTensorShape(name);
    void* ptr;
    if (engine->getTensorLocation(name) == TensorLocation::kDEVICE)
    {
        if (extent.d contains a -1)
        {
            auto allocator = std::make_unique<MyOutputAllocator>();
            context->setOutputAllocator(name, allocator.get());
            allocatorMap.emplace(name, std::move(allocator));
        }
        else
        {
            ptr = allocate device memory per extent and format
        }
    }
    else
    {
        ptr = allocate cpu memory per extent and format
    }
    context->setTensorAddress(name, ptr);
}
```

아래는 `reallocateOutput`을 구현하는 대표적인 방법이다.

1. 사이즈를 알 때까지 할당을 연기한다. `IExecutionContext::setTensorAddress` 호출하거나 텐서 주소에 `nulllptr`로 호출하면 안된다.
2. `IExecutionContext::getMaxOutputSize`를 통해 메모리 상한 크기를 얻어서 충분한 메모리를 미리 할당한다. 충분한 메모리로 인해 엔진이 정상적으로 실행될 수 있지만, 상한값이 너무 높으면 문제가 될 수 있다.
3. 경험적으로 충분한 메모리를 미리 할당하고, `IExecutionContext::setTensorAddress`를 사용하여 TensorRT에게 이를 알려준다. 만약 크기가 맞지 않으면 `reallocateOutput`이 `nullptr`을 리턴하도록 하여 엔진이 정상적으로 실패하도록 한다.
4. C에서와 같이 메모리를 사전에 할당하지만, 크기가 맞지 않으면 더 큰 버퍼를 가리키는 포인터를 리턴하도록 한다. 이렇게 하면 필요에 따라 버퍼가 증가한다.
5. 1번의 경우와 같이 크기를 알 때까지 할당을 연기한다. 그런 다음 더 큰 버퍼가 요청될 때까지 해당 메모리를 재사용하고 만약 더 큰 메모리가 필요할 때 4번과 같이 버퍼 크기를 증가시킨다.

5번 방법에 대한 구현 예시는 다음과 같다.
```c++
class FancyOutputAllocator : nvinfer1::IOutputAllocator
{
public:
    void reallocateOutput(
        char const* tensorName, void* currentMemory,
        uint64_t size, uint64_t alignment) override
    {
        if (size > outputSize)
        {
            // Need to reallocate
            cudaFree(outputPtr);
            outputPtr = nullptr;
            outputSize = 0;
            if (cudaMalloc(&outputPtr, size) == cudaSuccess)
            {
                outputSize = size;
            }
        }
        // If the cudaMalloc fails, outputPtr=nullptr, and engine
        // gracefully fails.
        return outputPtr;
    }

    void notifyShape(char const* tensorName, Dims const& dims)
    {
        // Remember output dimensions for later use.
        outputDims = dims;
    }

    // Saved dimensions of the output tensor
    Dims outputDims{};

    // nullptr if memory could not be allocated
    void* outputPtr{nullptr};

    // Size of allocation pointed to by output
    uint64_t outputSize{0};

    ~FancyOutputAllocator() override
    {
        cudaFree(outputPtr);
    }
};
```

## Looking up Binding Indices for Multiple Optimization Profiles

이번 섹션에서 언급하는 내용은 `enqueueV2()`(deprecated) 대신 `enqueueV3()`를 사용한다면 스킵해도 된다. `IExecuteContext::setTensorAddress()`와 같이 텐서의 이름을 기반으로 하는 메소드들은 profile suffix가 필요하지 않다.

여러 profiles로부터 엕진을 빌드할 때, 각 profiles마다 별도의 binding indices가 있다. `K`번째 profile에 대한 I/O 텐서의 이름에는 `[profile K]`라는 이름이 뒤에 붙는다. 만약 `INetworkDefinition`이 `foo`라는 이름의 I/O 텐서를 갖고, `bindingIndex`가 인덱스가 `3`인 optimization profile에서의 텐서를 가리킬 때, `engine->getBindingName(bindingIndex)`는 `"foo [profile 3]"`을 리턴한다.

마찬가지로 만약 `ICudaEngine::getBindingIndex(name)`을 사용하여 profile `K`에 대한 인덱스를 얻으려면 지정한 텐서 이름에 `"[profile K]"`를 붙여주어야 한다. 예를 들어, `INetworkDefinition`에서 텐서의 이름이 `"foo"`라면, `engine->getBindingIndex("foo [profile 3]")`로 호출해야 optimization profile `3`의 `"foo"` 텐서의 binding index를 얻을 수 있다.

`K=0`에 대한 suffix는 항상 생략된다.

## Binding For Multiple Optimization Profiles

이번 섹션에서 설명하는 내용도 `euqueueV2()`를 사용하는 경우에 해당하는 내용이다. 새로운 인터페이스인 `enqueueV3()`는 binding indices를 사용하지 않는다.

네트워크가 4개의 입력과 1개의 출력을 사용하고, 3개의 optimization profiles로 빌드되었다면, 이 엔진에는 각 optimization profile당 5개의 바인딩을 가지게 되어 총 15개의 바인딩을 가진다.

<img src="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/opt-profile.PNG" height=150px style="display: block; margin: 0 auto; background-color:white"/>

테이블로 나타내면 위와 같다. 행은 profile이며, 테이블에 숫자는 바인딩 인덱스를 나타낸다. 첫 번째 profile의 바인딩 인덱스는 0~4, 두 번째 profile은 5~9, 세 번째 profile의 바인디이 인덱스는 10~14이다.

첫 번째 profile에 대한 바인딩을 의도했지만 다른 profile이 지정된 경우를 위한 "auto-correct" 인터페이스가 있다. 이 경우, TensorRT는 경고를 출력하고 올바른 바인딩 인덱스를 선택한다.

<br>

# Layer Extensions For Dynamic Shapes

일부 레이어에는 dynamic shape 정보를 지정할 수 있는 선택적 입력을 가진다. `IShapeLayer`는 런타임 시, 텐서의 shape에 액세스하는데 사용될 수 있다.

`IShapeLayer`는 입력 텐서의 차원을 포함하는 1D 텐서를 출력한다. 예를 들어, 입력 텐서의 차원이 `[2, 3, 5, 7]`이라면, 출력 텐서는 4개의 요소의 1D 텐서(`{2, 3, 5, 7}`)이다. 만약 입력 텐서가 스칼라라면, 이 텐서의 차원은 `[]`이고, 출력 텐서는 zero-element 1D 텐서(`{}`)이다.

`IResizeLayer`는 선택적으로 출력으로 원하는 차원 크기를 요소 값으로 갖는 두 번째 optional input을 받는다.

`IShuffleLayer`는 second transpose를 적용하기 전에 reshape 차원 크기를 요소로 갖는 optional second input을 받는다. 예를 들어, 아래 코드는 `Y` 텐서를 `X`의 차원과 갖도록 reshape 한다.
```c++
auto* reshape = network->addShuffle(Y);
reshape->setInput(1, network->addShape(X)->getOutput(0));
```

`ISliceLayer`는 start, size, stride에 해당하는 optional second, third, fourth 입력을 받는다.

아래 나열된 레이어들은 shapes를 계산하고 새로운 shape 텐서를 만드는데 사용될 수 있다.

- `IConcatenationLayer`
- `IElementWiseLayer`
- `IGatherLayer`
- `IIdentityLayer`
- `IReduceLayer`

<br>

# Restrictions For Dynamic Shapes

레이어의 weights는 고정된 크기를 갖기 때문에 레이어에는 다음의 제약이 있다.

- `IConvolutionLayer`와 `IDeconvolutionLayer`의 channel 차원은 빌드 시 상수이어야 한다.
- `IFullyConnectedLayer`의 마지막 3개의 차원은 build-time constant 이어야 한다.
- `Int8`에서 channel 차원은 build-time constant 이어야 한다.
- 추가적인 shape inputs를 받는 레이어(`IResizeLayer`, `IShuffleLayer`, `ISliceLayer`)에서 shape inputs은 optimization profile의 minimum/maximum 차원과 호환되어야 한다. 그렇지 않으면 빌드 에러 또는 런타임 에러가 발생할 수 있다.

<br>

# Execution Tensors Versus Shape Tensors

TensorRT 8.5에서는 execution 텐서와 shape 텐서의 구분이 상당 부분 사라졌다. 그러나 네트워크를 설계하거나 성능을 분석하는 경우에 내부를 이해하거나 어디서 동기화가 발생하는지 이해하는 것이 도움이 될 수 있다.

Dynamic shapes를 사용하는 엔진은 **ping-pong** execution 전략을 사용한다.

1. Compute the shapes of tensors on the CPU until a shape requiring GPU results is reached.
2. Stream work to the GPU until out of work or an unknown shape is reached If the latter, synchronize and go back to step 1.

Execution 텐서는 일반적은 TensorRT의 텐서이다. *Shape Tensor*는 shape 계산과 관련된 텐서이다. Shape 텐서는 반드시 `Int32`, `Float`, 또는 `Bool` 타입이어야 하고, 텐서의 shape는 빌드 시간에 결정되어야 하며, 이 텐서의 요소의 수는 64개를 넘을 수 없다. 네트워크의 I/O에서 shape 텐서에 대한 추가적인 제약 사항은 [Shape Tensor I/O](#shape-tensor-io-advanced)를 참조하면 된다.

예를 들어, 출력으로 입력 텐서의 차원을 나타내는 1D 텐서를 갖는 `IShapeLayer`가 있다. `IShuffleLayer`는 reshaping dimensions를 지정할 수 있는 optional second input을 받으며, 이는 shape tensor이어야 한다.

일부 레이어는 이들이 다룰 수 있는 텐서의 종류에 대해 "다형성(polymorphic)"을 갖는다. 예를 들어, `IElementWiseLayer`는 `INT32` execution 텐서 2개를 더하거나 `INT32` shape 텐서 2개를 더할 수 있다. 텐서의 타입은 최종 사용처에 따라 다르다. 만약 다른 텐서를 reshape하기 위해 사용되었다면, shape 텐서가 된다.

TensorRT에 shape 텐서가 필요하지만 해당 텐서가 execution 텐서로 분류된 경우, 런타임은 이 텐서를 GPU에서 CPU로 복사해야 하며 synchronization overhead가 발생된다.

## Formal Inference Rules

텐서를 분류하기 위해 TensorRT에서 사용되는 공식적인 inference rules은 type-inference algebra를 기반으로 한다. 아래에서 `E`는 execution 텐서를 나타내며 `S`는 shape 텐서를 나타낸다.

`IActivationLayer`는 execution 텐서를 입력으로 받고 execution 텐서를 출력으로 내보내기 때문에 다음과 같다.
```
IActivationLayer: E -> E
```

`IElementWiseLayer`는 각 텐서에 대해 다형성을 가지므로, 다음과 같다.
```
IElementWiseLayer: S × S -> S, E × E -> E
```

간결함을 위해 `t`는 텐서의 클래스를 나타내는 변수이고 모든 `t`는 동일한 텐서의 클래스를 지칭한다는 규칙을 채택하면, `IElementWiseLayer`는 다음과 같이 표현할 수 있다.
```
IElementWiseLayer: t × t -> t
```

두 번째 입력을 받는 `IShuffleLayer`는 두 번째 입력으로 shape 텐서를 받으며, 첫 번째 입력에 대해 다형성이다.
```
IShuffleLayer (two input): t × S -> t
```

`IConstantLayer`는 어떠한 입력도 받지 않지만, 두 종류의 텐서를 모두 생성할 수 있다.
```
IConstantLayer: -> t
```

`IShapeLayer`는 4개의 모든 조합 `E->E`, `E->S`, `S->E`, `S->S`이 가능하다. 따라서, 다음과 같이 표현한다.
```
IShapeLayer: t1 -> t2
```

shape 텐서를 조작하는데 사용될 수 있는 다른 레이어들에 대한 규칙들은 다음과 같다.
```
IAssertionLayer: S → 
IConcatenationLayer: t × t × ...→ t
IIfConditionalInputLayer: t → t
IIfConditionalOutputLayer: t → t
IConstantLayer: → t
IActivationLayer: t → t
IElementWiseLayer: t × t → t
IFillLayer: S → t
IFillLayer: S × E × E → E 
IGatherLayer: t × t → t
IIdentityLayer: t → t
IReduceLayer: t → t
IResizeLayer (one input): E → E
IResizeLayer (two inputs): E × S → E
ISelectLayer: t × t × t → t
IShapeLayer: t1 → t2
IShuffleLayer (one input): t → t
IShuffleLayer (two inputs): t × S → t
ISliceLayer (one input): t → t
ISliceLayer (two inputs): t × S → t
ISliceLayer (three inputs): t × S × S → t
ISliceLayer (four inputs): t × S × S × S → t
IUnaryLayer: t → t
all other layers: E × ... → E × ...
```

레이어의 출력은 이후에 둘 이상의 다른 레이어의 입력이 될 수 있으므로 추론된 타입은 배타적이지 않다. 예를 들어, `IConstantLayer`는 execution 텐서가 필요한 레이어와 shape 텐서가 필요한 레이어로 각각 전달될 수 있다. `IConstantLayer`의 출력은 두 타입으로 모두 분류되며 two-phase execution(?)에서 모두 사용될 수 있다.

Shape 텐서의 크기는 빌드 시간에 결정되어야 한다는 요구 사항으로 인해 `ISliceLayer`를 사용하여 shape 텐서를 조작하는 방법이 제한된다. 특히, 출력 텐서의 크기를 지정하는 세 번째 매개변수가 빌드 시 상수가 아닌 경우, 빌드할 때 출력 텐서의 길이를 알 수 없으므로 shape 텐서가 일정한 모양을 갖는다는 제한이 사라진다. 이 경우, 슬라이스는 여전히 동작하지만 텐서는 추가적인 shape 계산을 수행하기 위해 CPU로 다시 복사되어야 하는 execution 텐서로 간주되어 런타임에서 synchronization overhead가 발생한다.

모든 텐서의 랭크(rank)는 빌드 시간에 정해져야 한다. 예를 들어, `ISliceLayer`의 출력이 길이를 알 수 없는 1D 텐서이고, 이 텐서가 `IShuffleLayer`의 reshape 차원으로 사용되는 경우, shuffle의 출력은 빌드 시 랭크를 알 수 없기 때문에 이러한 패턴은 금지된다.

TensorRT의 추론은 `ITensor::isShapeTensor()`와 `ITensor::isExecutionTensor()` 메소드를 사용하여 텐서 타입을 알 수 있다. 텐서의 용도에 따라서 메소드의 리턴값이 달라질 수 있기 때문에 이러한 메소드를 호출하기 전에 먼저 네트워크 전체를 빌드해야 한다.

예를 들어, 부분적으로 빌드된 네트쿼크가 두 개의 텐서 T1과 T2를 더해 T3를 생성하고, 아직 shape 텐서로 필요한 것이 없는 경우에 `isShapeTensor()`는 3개의 텐서에 대해 모두 `false`를 반환한다. 만약 `IShuffleLayer`의 두 번째 입력으로 T3을 사용하면, `IShuffleLayer`의 두 번째 입력은 shape 텐서이어야 하고 `IElementWiseLayer`의 출력 텐서의 shape라면 해당 입력도 shape 텐서가 되어야 하기 때문에 3개의 텐서 모두 shape 텐서가 된다.

<br>

# Shape Tensor I/O (Advanced)

때때로 shape 텐서를 네트워크 I/O로 사용할 필요가 있다. 예를 들어, `IShuffleLayer`로만 구성된 레이어를 생각해보자. TensorRT는 해당 레이어의 두 번째 입력이 shape 텐서라고 추론한다. `ITensor::isShapeTensor()`는 `true`를 반환한다. 이는 input shape 텐서이므로 TensorRT에는 야래의 두 가지가 필요하다.

- At build time: the opotimization profile values of the shape tensor
- At run time  : the values of shape tensor

Input shape 텐서의 shape는 빌드시 항상 알려져 있다. 이는 Execution 텐서의 차원을 지정하는데 사용될 수 있기 때문이다.

Optimization profile 값은 `IOptimizationProfile::setShapeValues()`를 사용하여 설정할 수 있다. Runtime dimensions의 execution 텐서에 min/max/opt optimization dimensions를 제공하는 방식과 유사하게 shape 텐서에도 이 값들을 제공해야 한다.

Execution 텐서인지 shape 텐서인지에 대한 추론은 궁극적인 사용처에 따라 다르기 때문에 TensorRT는 네트워크의 출력이 shape 텐서인지 추론할 수 없다. `INetworkDefinition::markOutputForShapes()` 메소드를 사용하여 이를 알려주어야 한다.

디버깅을 위해 shape 정보를 출력하는 것 이외에도 이 기능은 엔진을 합성하는데 유용하다. 예를 들어, A에서 B로 또는 B에서 C로 연결하는 데 shape 텐서가 필요한 sub-network A, B, C에 각각 엔진을 빌드하는 상황을 생각해보자. 빌드는 C, B, A 역순으로 수행한다. 네트워크 C를 구축한 후, `ITensor::isShapeTensor()`를 사용하여 입력이 shape 텐서인지 확인하고 `INetworkDefinition::markOutputForShapes()`를 사용하여 네트워크 B에서 해당 출력 텐서를 마킹한다. 그런 다음 B의 어떤 입력이 shpae 텐서인지 확인하고 네트워크 A의 해당 텐서와 대응하는 출력을 마킹한다. (잘 이해가 되지 않는 부분...)

네트워크 경계에서의 shape 텐서는 반드시 `Int32` 타입이어야 하며, `Float`나 `Bool` 타입이 될 수 없다. `Bool`의 경우 0과 1을 갖는 I/O 텐서에 `INT8`을 사용하고 `IIdentityLayer`를 사용하여 `Bool`로 변환할 수 있다.

런타임에서 텐서가 I/O shape 텐서인지 확인하려면 `ICudnEngine::isShapeInferenceIO()` 메소드를 사용하면 된다.

<br>

# INT8 Calibration with Dynamic Shapes

Dynamic shape를 갖는 네트워크에 대해 INT8 calibration을 실행하려면, calibration optimization profile이 반드시 설정되어야 한다. Calibration은 profile의 kOPT 값을 사용하여 수행된다. Calibraiton input data sizee는 반드시 이 profile과 일치해야 한다.

Calibration optimization profile을 생성하려면 먼저 `IOptimizationProfile`을 일반 optimization profile과 동일한 방식으로 구성한다. 그런 다음 해당 profile을 build configuration에 설정한다.
```c++
config->setCalibrationProfile(profile);
```

Calibration profile은 유효하거나 `nullptr`이어야 한다. `kMIN`과 `kMAX` 값은 `kOPT`로 덮어써진다. 현재 calibration profile을 체크하려면 `IBuilderConfig::getCalibrationProfile`을 사용하면 된다. 이 메소드는 현재 calibration profile의 포인터를 반환하거나 profile이 unset되었다면 `nullptr`을 반환한다. Dynamic shape 네트워크에 대해 calibration을 수행할 때, Calibrator의 `getBatchSize()` 메소드는 반드시 `1`을 반환해야 한다.

> 만약 calibration optimization profile이 설정되지 않았다면, 첫 번째 네트워크 optimization profile이 calibration optimization profile로 사용된다.

<br>

# References

- [NVIDIA TensorRT Documentation: Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)