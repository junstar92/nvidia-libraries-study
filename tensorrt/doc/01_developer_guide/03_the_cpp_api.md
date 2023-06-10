# Table of Contents

- [Table of Contents](#table-of-contents)
- [The C++ API](#the-c-api)
- [The Build Phase](#the-build-phase)
  - [Creating a Network Definition](#creating-a-network-definition)
  - [Importing a Model Using the ONNX Parser](#importing-a-model-using-the-onnx-parser)
  - [Building an Engine](#building-an-engine)
- [Deserializing a Plan](#deserializing-a-plan)
- [Performing Inference](#performing-inference)
- [References](#references)

<br>

# The C++ API

이번 섹션에서는 TensorRT C++ API의 기본적인 사용 방법에 대해서 다룬다. 여기서 사용되는 예제 코드는 [sampleOnnxMNIST](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST)를 기반이다.

> 아래 포스팅에서 Network Definition API를 사용하는 것과 ONNX Parser를 사용하는 것에 대한 실제 구현을 살펴볼 수 있다.
> - [MNIST Classification using Network Definition APIs](/tensorrt/study/01_mnist_cnn_api.md)
> - [MNIST Classification using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)

C++ API는 `NvInfer.h` 헤더 파일을 통해 액세스할 수 있고, 이 헤더의 네임스페이스는 `nvinfer1`이다.
```c++
#include "NvInfer.h"

using namespace nvinfer1;
```

TensorRT에서의 인터페이스 클래스는 `I`라는 접두사를 가진다. 예를 들어, `ILogger`, `IBuilder`, 등이 있다.

CUDA context는 TensorRT가 처음 CUDA를 호출할 때 자동으로 생성된다. 일반적으로 TensorRT를 처음 호출하기 전에 CUDA Context를 직접 만들고 구성하는 것이 좋다.

> 이 섹션에서 객체의 lifetime을 설명하기 위해 스마트 포인터를 사용하지 않는다. 하지만, 실사용에서는 TensorRT 인터페이스와 함께 사용하는 것이 좋다.

<br>

# The Build Phase

Builder를 생성하려면, 먼저 `ILogger` 인터페이스를 인스턴스화해야 한다. 아래 예제 코드에서는 모든 경고 메세지(에러 포함)을 출력하고, 정보성 메세지는 무시하는 로거를 구현하여 사용한다.
```c++
class Logger : puglic ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;
```

그런 다음, 아래 예제 코드와 같이 `logger`를 전달하여 `builder`를 생성할 수 있다.
```c++
IBuilder* builder = createInferBuilder(logger);
```

## Creating a Network Definition

Builder가 생성되고 난 후, 모델을 최적화하기 위한 첫 번째 단계는 network definition을 생성하는 것이다.
```c++
uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

INetworkDefinition* network = builder->createNetworkV2(flag);
```

위 예제 코드에서 `kEXPLICIT_BATCH` 플래그는 ONNX parser를 사용하여 모델을 import하는 경우, 필수로 설정해주어야 한다. 이와 관련된 내용은 [Explicit Versus IMplicit Batch](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)에서 살펴볼 수 있다.

## Importing a Model Using the ONNX Parser

생성된 network definition을 채워주어야 하는데, 여기서는 ONNX representation으로부터 채운다. ONNX parser API는 `NvOnnxParser.h` 헤더를 통해 액세스할 수 있고, 네임스페이스는 `nvonnxparser`이다.
```c++
#include "NvOnnxParser.h"

using namespace nvonnxparser;
```

네트워크를 채우기 위한 ONNX parser는 아래와 같이 생성할 수 있다.
```c++
IParser* parser = createParser(*network, logger);
```
그런 다음, ONNX 파일을 읽어준다.
```c++
parser->parseFromFile(modelFile,
    static_cast<int32_t>(ILogger::Severity::kWARNING));
for (int32_t i = 0; i < parser.getNbErrors(); i++) {
    std::cout << parser->getError(i)->desc() << std::endl;
}
```

여기서 중요한 점은 TensorRT network definition이 model weights에 대한 포인터를 포함하고 있다는 것이며, 이는 builder에 의해 최적화된 엔진으로 복사된다. 위 코드에서는 parser를 사용하여 네트워크가 생성되었기 때문에 parser가 네트워크의 weight에 대한 메모리를 소유하고 있다. 따라서, builder가 실행될 때까지 parser 객체는 유지되어야 한다.

## Building an Engine

다음 단계는 TensorRT가 모델을 최적화하는 방법을 지정해주기 위해서 build configuration을 생성하는 것이다.
```c++
IBuilderConfig* config = builder->createBuilderConfig();
```

`IBuilderConfig`에는 TensorRT가 네트워크를 최적화하는 방법을 제어하기 위한 많은 속성들이 있다.

한 가지 중요한 속성 중 하나는 maximum workspace size 이다. 각 레이어 구현에서는 종종 임시 공간이 필요하며, 이 속성은 네트워크의 모든 레이어가 사용할 수 있는 최대 크기를 제한한다. 이 공간이 충분하지 않다면 TensorRT가 최적화된 레이어의 구현을 찾지 못할 수도 있다. 따로 지정하지 않는다면, 기본적으로 GPU device의 총 global memory 크기로 설정된다. 이 속성은 단일 GPU 장치에 여러 엔진을 빌드해야 하는 경우에 필요할 수 있다. 최대 크기는 아래와 같이 지정한다.
```c++
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20); // 1 MB
```

<br>

설정이 모두 지정되면, 아래의 API 호출을 통해 엔진을 빌드할 수 있다.
```c++
IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
```

생성된 **serialized engine**에는 필요한 weight의 복사본이 포함되어 있으므로, parser, network definition, builder configuration 및 builder는 더 이상 필요하지 않으며 삭제해도 된다.
```c++
delete parser;
delete network;
delete config;
delete builder;
```

생성된 serialized engine을 파일로 저장하고 나면, 이 또한 삭제할 수 있다.
```c++
delete serializedModel;
```

> 빌드 과정을 통해 생성된 엔진은 직렬화된 데이터(serialized data)이다. 이러한 데이터를 TensorRT에서 플랜(Plan)이라고 부른다. 추론을 위해서는 플랜을 역직렬화하는 과정이 필요하다. 이는 바로 아래 섹션([Deserializing a Plan](#deserializing-a-plan))에서 다룬다.

> Serialized Engine은 플랫폼이나 TensorRT 버전 간 호환되지 않는다. 엔진은 엔진이 빌드된 정확한 GPU 모델에 따라서도 다르다.

> 엔진을 빌드하는 것은 offline process를 위한 것이므로 상당한 시간이 소요될 수 있다. 만약 빌드 시간을 단축하고 싶다면, [Optimizing Builder Performance](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt-builder-perf)를 참조하길 바란다.

<br>

# Deserializing a Plan

이전까지의 과정을 통해 최적화된 모델을 직렬화(serialization)했고, 이를 사용하여 추론을 수행하려면 먼저 `Runtime` 인터페이스 인스턴스를 생성해야 한다. Builder와 같이 Runtime은 logger 인스턴스가 필요하다.
```c++
IRuntime* runtime = createInferRuntime(logger);
```

그런 다음, 모델(serialized data)을 버퍼로 읽고 역직렬화하여 엔진을 얻을 수 있다. 여기서 모델(serialized data)는 이전 빌드 단계에서 저장한 serialized engine을 의미한다.
```c++
ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize);
```

<br>

# Performing Inference

엔진은 최적화된 모델을 hold하고 있다. 하지만 추론을 수행하려면 중간 활성화(intermediate activations)를 위한 추가 상태를 관리해야 한다. 이는 `ExecutionContext` 인터페이스를 사용하여 수행된다.
```c++
IExecutionContext* context = engine->createExecutionContext();
```

엔진은 여러 개의 execution context를 가질 수 있다. 따라서, 하나의 weight 세트만 사용하여 여러 개의 중첩된 추론이 가능하다.

> 이에 대한 예외가 있다고 아래와 같이 언급하고 있다.
> 
> A current exception to this is when using dynamic shapes, when each optimization profile can only have one execution context, unless the preview feature, `kPROFILE_SHARING_0806`, is specified.

추론을 수행하려면, input과 output에 대한 버퍼를 TensorRT에 전달해야 한다. 이는 ExecutionContext의 API `setTensorAddress`를 호출하여 전달한다. 이 API는 텐서의 이름과 버퍼의 주소를 매개변수로 받으며, input/output 텐서의 이름은 엔진을 쿼리하여 알아낼 수 있다.
```c++
context->setTensorAddress(INPUT_NAME, inputBuffer);
context->setTensorAddress(OUTPUT_NAME, outputBuffer);
```

그런 다음, `enqueueV3()` 메소드를 호출하여 추론을 수행할 수 있다. `enqueueV3()` 메소드는 CUDA Stream을 매개변수로 받으며, 비동기 실행이 가능하다. 이외에도 `executeV2()` 메소드도 있는데, 이는 추론을 동기식으로 수행한다.
```c++
context->enqueueV3(stream);
```

다만, 네트워크는 네트워크의 구조와 기능에 따라서 비동기식으로 실행될 수도 있고 아닐 수도 있다. 데이터의 종속성, DLA 사용 유무, loops 및 동기식으로 동작하는 plugins에 따라서 강제로 동기식으로 동작할 수 있다.

<br>

# References

- [NVIDIA TensorRT Documentation: The C++ API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics)
- [Sample: sampleOnnxMNIST](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST)