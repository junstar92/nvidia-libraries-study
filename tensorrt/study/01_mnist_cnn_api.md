# Table of Contents

- [Table of Contents](#table-of-contents)
- [MNIST Classification using Network Definition APIs](#mnist-classification-using-network-definition-apis)
- [Logger](#logger)
- [Build Phase](#build-phase)
  - [Creating a Network Definition](#creating-a-network-definition)
  - [Building an Engine](#building-an-engine)
  - [Saving a Serialized Engine](#saving-a-serialized-engine)
  - [Build Log](#build-log)
- [Deserializing a Plan](#deserializing-a-plan)
- [Performing Inference](#performing-inference)
- [References](#references)

<br>

# MNIST Classification using Network Definition APIs

이번 포스팅에서는 TensorRT의 Network Definition API를 사용하여 네트워크를 구축하고, 빌드 및 추론까지의 과정을 c++로 어떻게 구현할 수 있는지 살펴본다. 이 과정에서 빌드된 엔진(plan)을 local memory에 저장하고 이를 읽어서 엔진을 역직렬화(deserialization)하는 방법도 살펴본다.

MNIST 데이터를 사용하여 0부터 9까지의 숫자를 분류하는 간단한 CNN 모델을 사용한다. PyTorch를 통해 해당 모델을 학습했으며, 레이어의 weights를 바이너리 파일로 추출하여 TensorRT로 빌드할 때 사용한다. 학습 코드는 아래 경로의 주피터 노트북에서 확인할 수 있으며, 이로부터 추출된 weights 경로와 샘플 데이터의 바이너리 파일 또한 아래에 나열하였다.

- [Jupyter Notebook: Training simple mnist CNN model](/cudnn/code/mnist_cnn_v7/mnist-cnn.ipynb)
- [Weights of the Network](/cudnn/code/mnist_cnn_v7/params/)
- [Sample Mnist Data](/cudnn/code/mnist_cnn_v7/digits/) (0 ~ 9)

이 포스팅에서 사용한 전체 코드는 아래 링크에서 확인할 수 있다.

- [mnist_cnn_api.cpp](/tensorrt/code/mnist_cnn_api/mnist_cnn_api.cpp)

<br>

# Logger

TensorRT에서 인터페이스 인스턴스를 생성할 때, `ILogger` 인스턴스를 인자로 전달해주어야 한다. 로거는 TensorRT 내부에서 출력하는 에러, 경고, 정보 등을 처리하기 위한 클래스이며, 사용자가 TensorRT API에서 제공하는 `ILogger` 클래스를 상속받아서 구현을 채워주어야 한다. 여기서는 아래와 같이 `Logger` 클래스를 구현했으며, 클래스 인스턴스를 생성할 때 아무런 인자로 전달하지 않으면 기본적으로 에러와 경고만 출력하게 된다.

```c++
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
```

<br>

# Build Phase

빌드를 위한 `IBuilder` 인스턴스를 생성한다. 이때, 위에서 생성해둔 `gLogger` 인스턴스를 인자로 전달한다.
```c++
IBuilder* builder = createInferBuilder(gLogger);
```

## Creating a Network Definition

이 포스팅에서는 ONNX가 아닌 직접 TensorRT의 Network Definition API를 사용하여 네트워크를 구성한다. 이를 위한 첫 번째 단계는 `INetworkDefinition` 인스턴스를 생성하는 것이다. 이때, flag를 통해 `kEXPLICIT_BATCH`로 설정한다. 이에 대한 내용은 [Explicit versus Implicit Batch](/tensorrt/doc/01_developer_guide/06-07_explicit_versus_implicit_batch.md)에서 자세히 다루고 있다.

```c++
// create network
uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
INetworkDefinition* network = builder->createNetworkV2(flag);
```

그리고, 생성한 `network` 인스턴스를 통해 우리가 빌드할 네트워크 레이어들을 구성한다. 코드에서는 `createNetwork()` 함수를 통해 네트워크를 구성한다.

`createNetwork()`의 구현을 보면 우선 필요한 weights를 준비해주고 있다. 각 레이어에 필요한 weights는 `nvinfer1::Weights`라는 클래스로 전달되어야 한다. 그리고, `Weights` 클래스는 해당 weight의 데이터 타입(`type`), 요소의 갯수(`count`), 그리고 해당 데이터를 가리키는 포인터(`value`)를 멤버 변수로 가진다. 우리가 구현할 네트워크에서 weight를 갖는 레이어는 `conv1`, `conv2`, `fc` 레이어라서 아래와 같이 총 6개의 `Weights`를 준비한다.
```c++
conv1KernelWeight.count = 32 * 3 * 3;
conv1KernelWeight.type = DataType::kFLOAT;
conv1KernelWeight.values = conv1_weight;
conv1BiasWeight.count = 32;
conv1BiasWeight.type = DataType::kFLOAT;
conv1BiasWeight.values = conv1_bias;

conv2KernelWeight.count = 64 * 32 * 3 * 3;
conv2KernelWeight.type = DataType::kFLOAT;
conv2KernelWeight.values = conv2_weight;
conv2BiasWeight.count = 64;
conv2BiasWeight.type = DataType::kFLOAT;
conv2BiasWeight.values = conv2_bias;

fcKernelWeight.count = 7 * 7 * 64 * 10;
fcKernelWeight.type = DataType::kFLOAT;
fcKernelWeight.values = fc_weight;
fcBiasWeight.count = 10;
fcBiasWeight.type = DataType::kFLOAT;
fcBiasWeight.values = fc_bias;
```
이때, `value` 포인터가 가리키는 것은 실제 weights의 값을 가지고 있는 주소이며, 이 주소는 host memory에 있는 주소이어야 한다. 해당 `Weights`가 전달된 레이어는 이 포인터에 대해 deep copy를 수행하지 않기 때문에 serialized engine 생성이 완료될 때까지 해당 weights에 대한 메모리는 유지되어야 한다.

이렇게 준비가 완료되면, 이제 네트워크를 구성한다.
```c++
// conv1 - relu1 - maxpool1
auto conv1 = network->addConvolution(*input, 32, DimsHW{3,3}, conv1KernelWeight, conv1BiasWeight);
conv1->setStrideNd(DimsHW{1,1});
conv1->setPaddingNd(DimsHW{1,1});
conv1->setDilationNd(DimsHW{1,1});
conv1->setNbGroups(1);
conv1->setName("conv1");

auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
relu1->setName("relu1");

auto maxpool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2,2});
maxpool1->setStrideNd(DimsHW{2,2});
maxpool1->setName("maxpool1");

// conv2 - relu2 - maxpool2
auto conv2 = network->addConvolution(*maxpool1->getOutput(0), 64, DimsHW{3,3}, conv2KernelWeight, conv2BiasWeight);
conv2->setStrideNd(DimsHW{1,1});
conv2->setPaddingNd(DimsHW{1,1});
conv2->setDilationNd(DimsHW{1,1});
conv2->setNbGroups(1);
conv2->setName("conv2");

auto relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
relu2->setName("relu2");

auto maxpool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2,2});
maxpool2->setStrideNd(DimsHW{2,2});
maxpool2->setName("maxpool2");

// reshape
auto reshape = network->addShuffle(*maxpool2->getOutput(0));
reshape->setReshapeDimensions(Dims4{1, -1, 1, 1});
reshape->setName("reshape");

// fc
auto fc = network->addFullyConnected(*reshape->getOutput(0), 10, fcKernelWeight, fcBiasWeight);
fc->setName("fc");

// mark output
fc->getOutput(0)->setName("output");
network->markOutput(*fc->getOutput(0));
```

여기서 주목할 점은 네트워크의 입력은 `network` 인스턴스의 `addInput` 메소드로 추가한다는 것이고, 출력은 `markOutput` 메소드로 추가한다는 것이다. 그리고, 각 레이어의 입력으로 전달되는 인자의 타입은 `ITensor`이다.

이렇게 생성된 레이어를 `network` 인스턴스를 통해 쿼리해보면,
```c++
// network query
printf("- Network: %s[(%d,%d,%d,%d)] -> ", network->getInput(0)->getName(), network->getInput(0)->getDimensions().d[0],network->getInput(0)->getDimensions().d[1],network->getInput(0)->getDimensions().d[2],network->getInput(0)->getDimensions().d[3]);
int num_layers = network->getNbLayers();
for (int i = 0; i < num_layers; i++) {
    printf("%s[(%d,%d,%d,%d)] -> ", network->getLayer(i)->getName(), network->getLayer(i)->getOutput(0)->getDimensions().d[0],network->getLayer(i)->getOutput(0)->getDimensions().d[1],network->getLayer(i)->getOutput(0)->getDimensions().d[2],network->getLayer(i)->getOutput(0)->getDimensions().d[3]);
}
printf("%s\n", network->getOutput(0)->getName());
```
이 네트워크가 아래와 같이 구성되었다는 것을 확인할 수 있다.

`Network: input[(1,1,28,28)] -> conv1[(1,32,28,28)] -> relu1[(1,32,28,28)] -> maxpool1[(1,32,14,14)] -> conv2[(1,64,14,14)] -> relu2[(1,64,14,14)] -> maxpool2[(1,64,7,7)] -> reshape[(1,3136,1,1)] -> fc[(1,10,1,1)] -> output`


## Building an Engine

네트워크 구성이 끝나면, 이제 엔진을 빌드하기 위한 configuration을 설정한다.
```c++
// build configuration
IBuilderConfig* config = builder->createBuilderConfig();
config->clearFlag(BuilderFlag::kTF32);
//config->setFlag(BuilderFlag::kFP16);
```
코드에서는 default로 설정되는 `TF32` 플래그를 clear 했다. 이외의 플래그는 설정하지 않았기 때문에 해당 네트워크는 `FP32`로 빌드된다. 만약 빌드할 때, `FP16` 최적화도 포함하고 싶다면, 주석 처리되어 있는 코드를 활성화하면 된다. `FP16` 플래그를 활성화했다면, TensorRT는 네트워크를 최적화할 때, `FP32`와 `FP16`에 대한 구현을 모두 고려하게 되고, 가장 빠른 정밀도를 선택하게 된다. 경험적으로 거의 대부분 `FP16` 구현이 선택되는 것 같다. 하지만, 복잡한 네트워크에서 특정 레이어가 `FP32` 구현만 사용될 때, `formatting`에 걸리는 시간까지 고려하며, 그래도 `FP32`보다 빠르다면 `FP16` 구현을 선택하는 것 같다.

Build configuration 설정이 완료되었다면, 이제 아래의 API 호출을 통해 네트워크를 빌드한다.
```c++
// build serialized network
auto plan = builder->buildSerializedNetwork(*network, *config);
```
빌드가 완료된 후, serialized network, 즉, 플랜(plan) 바이너리 데이터를 리턴한다. 이때, 리턴되는 타입은 `IHostMemory`이다.

## Saving a Serialized Engine

이렇게 생성된 플랜을 재사용하려면 다음과 같이 local memory에 저장하면 된다.
```c++
// save serialized engine as plan in memory
std::ofstream ofs("plan.bin"); 
ofs.write(reinterpret_cast<const char*>(plan->data()), plan->size());
```

그리고, 플랜 생성까지 완료했고 얻은 플랜을 저장했다면 더 이상 위에서 사용한 TensorRT 인스턴스들은 필요없기 때문에 리소스를 해제해주면 된다.
```c++
delete network;
delete config;
delete builder;
delete plan;
```

## Build Log

Logger의 `Severity` 레벨에 따라 다르지만, `kVERBOSE`로 설정하면 모든 로그를 살펴볼 수 있다.
```
[I] [MemUsageChange] Init CUDA: CPU +331, GPU +0, now: CPU 336, GPU 4831 (MiB)
[I] [MemUsageChange] Init builder kernel library: CPU +328, GPU +104, now: CPU 683, GPU 4935 (MiB)
```
위와 같이 인스턴스를 생성할 때, 사용하는 메모리 크기도 알 수 있고,

빌드할 때, 아래와 같은 그래프 최적화가 어떻게 이루어졌는지도 살펴볼 수 있다. 우리가 구현한 네트워크에서는 conv->relu 과정이 하나의 연산으로 fusion된 것을 확인할 수 있다. 따라서, 그래프 최적화 이전에는 총 8개의 레이어로 구성되었지만, 그래프 최적화 이후에는 총 6개의 레이어가 되었다.
```
[V] Original: 8 layers
[V] After dead-layer removal: 8 layers
[V] After Myelin optimization: 8 layers
[V] Running: FCToConvTransform on fc
[V] Convert layer type of fc from FULLY_CONNECTED to CONVOLUTION
[V] Running: ShuffleErasure on shuffle_between_(Unnamed Layer* 6) [Shuffle]_output_and_fc
[V] Removing shuffle_between_(Unnamed Layer* 6) [Shuffle]_output_and_fc
[V] Applying ScaleNodes fusions.
[V] After scale fusion: 8 layers
[V] Running: ConvReluFusion on conv1
[V] ConvReluFusion: Fusing conv1 with relu1
[V] Running: ConvReluFusion on conv2
[V] ConvReluFusion: Fusing conv2 with relu2
[V] After dupe layer removal: 6 layers
[V] After final dead-layer removal: 6 layers
[V] After tensor merging: 6 layers
[V] After vertical fusions: 6 layers
[V] After dupe layer removal: 6 layers
[V] After final dead-layer removal: 6 layers
[V] After tensor merging: 6 layers
[V] After slice removal: 6 layers
[V] After concat removal: 6 layers
[V] Trying to split Reshape and strided tensor
[V] Graph construction and optimization completed in 0.00276015 seconds.
```

그래프 최적화 이후에는 최적의 구현을 선택하기 위한 kernel timing이 수행되고, 가장 빠른 커널이 선택된다. 빌드가 완료되면 아래와 같이 엔진의 레이어들에 대한 정보를 출력해준다.
```
[V] Engine Layer Information:
Layer(CaskConvolution): conv1 + relu1, Tactic: 0x4727434768e46395, input[Float(1,1,28,28)] -> (Unnamed Layer* 1) [Activation]_output[Float(1,32,28,28)]
Layer(TiledPooling): maxpool1, Tactic: 0x00000000005f0101, (Unnamed Layer* 1) [Activation]_output[Float(1,32,28,28)] -> (Unnamed Layer* 2) [Pooling]_output[Float(1,32,14,14)]
Layer(FusedConvActConvolution): conv2 + relu2, Tactic: 0x0000000000a2ffff, (Unnamed Layer* 2) [Pooling]_output[Float(1,32,14,14)] -> (Unnamed Layer* 4) [Activation]_output[Float(1,64,14,14)]
Layer(TiledPooling): maxpool2, Tactic: 0x00000000005e0101, (Unnamed Layer* 4) [Activation]_output[Float(1,64,14,14)] -> (Unnamed Layer* 5) [Pooling]_output[Float(1,64,7,7)]
Layer(NoOp): reshape, Tactic: 0x0000000000000000, (Unnamed Layer* 5) [Pooling]_output[Float(1,64,7,7)] -> (Unnamed Layer* 6) [Shuffle]_output[Float(1,3136,1,1)]
Layer(CublasConvolution): fc, Tactic: 0x0000000000000000, (Unnamed Layer* 6) [Shuffle]_output[Float(1,3136,1,1)] -> output[Float(1,10,1,1)]
```
위 로그에서 Tactic이 바로 선택된 커널이며, 각 레이어에서 입력과 출력의 차원 크기도 알 수 있다.

마지막에는 해당 엔진이 추론하기 위해 필요한 메모리의 크기도 나열해준다.
```
[V] Total per-runner device persistent memory is 0
[V] Total per-runner host persistent memory is 5328
[V] Allocated activation device memory of size 125952
```

<br>

# Deserializing a Plan

빌드한 플랜을 저장하고, 저장된 플랜을 다시 읽어서 역직렬화(deserialization)하는 과정은 다음과 같다.
```c++
/**** Runtime Phase ****/
ifs.open("plan.bin", std::ios_base::binary);
ifs >> std::noskipws;
std::string plan;
std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), std::back_inserter(plan));

IRuntime* runtime = createInferRuntime(gLogger);
ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
IExecutionContext* context = engine->createExecutionContext();
```

먼저 `IRuntime` 인터피에스 인스턴스를 생성하고, 런타임을 통해 플랜 데이터를 역직렬화한다. 성공적으로 역직렬화가 수행되면 `ICudaEngine`의 인스턴스를 리턴하게 된다.

그런 다음, 추론을 실제로 수행할 execution context를 `engine`으로부터 생성한다.

<br>

# Performing Inference

이제 생성한 `context`를 통해 어떻게 추론할 수 있는지 살펴보자.

추론하기 전에 아래 코드는 필요한 메모리를 할당한다.
```c++
// memory allocation for input, output
Dims input_dims = engine->getBindingDimensions(0);
Dims output_dims = engine->getBindingDimensions(1);

void *input, *output;
void *d_input, *d_output;

input = (void*)malloc(input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float));
output = (void*)malloc(output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * output_dims.d[3] * sizeof(float));
cudaMalloc(&d_input, input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float));
cudaMalloc(&d_output, output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * output_dims.d[3] * sizeof(float));
```

사용자 입장에서 엔진은 블랙박스라고 볼 수 있다. 즉, 우리에게 중요한 것은 어떤 입력이 엔진으로 들어가고, 어떤 출력이 엔진으로부터 나오는지가 중요하다. 레이어 중간 중간 과정에서 필요한 메모리는 사용자가 신경쓸 필요가 없다.

네트워크를 빌드할 때, 우리는 이미 입력 하나, 출력 하나라는 것을 알고 있다. 그래서 당연히 엔진에 바인딩된 텐서 중 0번 인덱스가 입력이고 1번 인덱스가 출력이라는 것이 자명하다. 확실하게 하고 싶다면, `engine`을 통해서 입력/출력의 갯수를 쿼리하고, 어떤 인덱스가 입력이고 출력인지 쿼리하면 된다. 여기에서는 입력과 출력이 단순하게 하나씩 존재하기 때문에 따로 쿼리하는 과정은 생략하였다.

엔진의 0번 인덱스 바인딩 텐서(입력)의 차원 크기와 1번 인덱스 바인딩 텐서(출력)의 차원 크기를 쿼리하고 이를 이용하여 host와 device 메모리를 각각 할당해주고 있다. 네트워크를 빌드할 때, 입력/출력 텐서의 데이터 타입을 별도로 지정하지 않았기 때문에 default인 `FP32` 타입이 사용된다. 따라서, `sizeof(float)`로 메모리가 할당된다.

> 참고로 바인딩되는 텐서의 데이터 타입과 빌드 시 선택하는 빌드 정밀도는 서로 독립적이다. 따라서, build config로 `FP16` 플래그를 활성화하더라도 내부적으로 `FP16` 연산이 수행된다는 것이지 엔진에 바인딩되는 텐서의 타입은 따로 설정해주지 않는 한 `FP32` 타입이다.

메모리가 준비되었으면, 이제 추론을 수행할 수 있다. 아래 코드에서는 추론 속도를 측정하기 위해 CUDA Event를 사용하고 있다.
```c++
cudaEvent_t start, stop;
float msec = 0.f;
cudaEventCreate(&start);
cudaEventCreate(&stop);

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
```

추론을 위해 `context->executeV2(binding)`을 호출한다. `executeV2()` API는 추론을 동기식으로 수행하는 메소드이다. 전달되는 포인터 배열 `binding`은 각 바인딩 인덱스에 맞는 위치에 해당 메모리 주소를 위치시켜주면 된다. 여기서는 0번 인덱스가 입력이고, 1번 인덱스가 출력이므로, 해당 순서와 동일하게 메모리를 위치한다.

출력 결과는 다음과 같다.
```
...
............................
............................
............................
............................
........**..................
......******................
.....********...............
......***..****.............
.....***....****............
.....**......****...........
.....**.......***...........
.....**...*...***...........
......*..*********..........
.........***********........
..........************......
..................*****.....
....................****....
.........**..........***....
.........***..........***...
..........**..........***...
..........****......*****...
...........*************....
.............*********......
................*...........
............................
............................
............................
............................
Digit: 3 (0.911075)
Elapsed Time: 0.031744 ms

............................
............................
............................
............................
............................
...........*................
...........*.......**.......
..........**........*.......
..........**........*.......
.........**........**.......
........**.........**.......
........**........***.......
.......**.........**........
.......**........***........
.......**........***........
.......**........**.........
.......***....*****.........
........***********.........
................***.........
.................**.........
................***.........
.................**.........
................***.........
................**..........
................*...........
............................
............................
............................
Digit: 4 (1)
Elapsed Time: 0.031744 ms
...
```

동일한 네트워크를 cuDNN으로 구현한 내용을 [Example: Mnist CNN (using legacy APIs](/cudnn/study/02_mnist_cnn(v7).md)에서 확인할 수 있는데, cuDNN 구현에서는 하나의 데이터를 추론하는데 약 0.06 ~ 0.09 ms의 시간이 걸렸던 것에 비해 TensorRT 구현이 약 0.03 ms로 조금 더 빠르다는 결과를 확인할 수 있다. 추론 결과, 확률 값은 모두 비슷하게 추론되었다.

<br>

# References

- [NVIDIA cuDNN Documentation: The C++ API](/tensorrt/doc/01_developer_guide/03_the_cpp_api.md)