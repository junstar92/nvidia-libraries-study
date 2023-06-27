# Table of Contents

- [Table of Contents](#table-of-contents)
- [MNIST Classification with Dynamic Shapes for Batch Inference](#mnist-classification-with-dynamic-shapes-for-batch-inference)
- [Exporting ONNX file with Dynamic Axes](#exporting-onnx-file-with-dynamic-axes)
- [Optimization Profiles](#optimization-profiles)
- [Input/Output Memory Allocation](#inputoutput-memory-allocation)
- [Performing Inference](#performing-inference)
- [References](#references)

<br>

# MNIST Classification with Dynamic Shapes for Batch Inference

이번 포스팅에서는 dynamic shapes를 사용한 MNIST Classification 구현에 대해서 살펴본다. Dynamic shape를 적용할 차원은 입력 차원 중에서도 첫 번째 차원이다. 즉, 효율적인 추론을 위한 batch inference를 구현하기 위해서 첫 번째 차원인 batch size를 dynamic shape로 적용한다. 구현 자체는 [Mnist CNN using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)와 유사하다. 단, 이번 구현에서는 TensorRT 8.5.1에서 도입된 `enqueueV3()` API를 사용한다는 점에서 다르다.

전체 구현 코드는 [mnist_cnn_onnx_dynamic.cpp](/tensorrt/code/mnist_cnn_onnx/mnist_cnn_onnx_dynamic.cpp)에서 확인할 수 있다.

# Exporting ONNX file with Dynamic Axes

[Mnist CNN using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)에서와 마찬가지로 PyTorch로 구현하여 학습한 모델을 ONNX로 추룰하여 TensorRT의 엔진을 빌드하는데 사용한다. 이때, PyTorch 모델을 ONNX로 추출할 때 아래와 같이 `dynamic_axes`를 설정해주어야 추출한 ONNX를 TensorRT에서 파싱할 때, 다른 작업없이 dynamic shapes를 사용할 수 있다. 만약, ONNX를 추출할 때, `dynamic_axes`를 지정하지 않는다면, 파싱한 `INetworkDefinition`에서 직접 입력 차원을 수정하여 dynamic shape로 수정해주어야 한다.
```python
# Export ONNX
dummy_input = torch.randn(1,1,28,28, device=device)
torch.onnx.export(
    model,
    dummy_input,
    "mnist_cnn_dynamic.onnx",
    input_names=["input"],
    output_names=["output"],
    verbose=True,
    dynamic_axes={
        "input": [0]
    })
```

> 학습 방법 및 ONNX 추출 코드는 [mnist-cnn-dynamic.ipynb](/tensorrt/code/mnist_cnn_onnx/notebook/mnist-cnn-dynamic.ipynb)에서 확인할 수 있음.

# Optimization Profiles

[Optimization profile](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md#optimization-profiles)는 auto-tuner가 최적화하는데 사용할 각 네트워크 입력 차원의 범위를 나타낸다. Runtime dimensions(dynamic shapes)를 사용할 때, 적어도 하나의 optimization profile을 빌드하기 전에 지정해주어야 한다. 구현은 다음과 같다.
```c++
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
```

Batch Inference가 목적이므로 추출한 ONN로부터 파싱한 네트워크의 입력 차원을 쿼리하면 `(-1, 1, 28, 28)`이다. 여기서 우리는 `-1`에 해당하는 배치 차원의 범위를 optimizer profile로 지정한다. 위 구현에서는 임의로 `min: 1, opt: 10, max: 64`로 지정하였다. 이 값은 어플리케이션에 따라 적절하게 지정해주면 된다. 설정한 optimizer profile를 config에 추가해주면 optimization profile 준비는 끝이다.

# Input/Output Memory Allocation

[Mnist CNN using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)에서는 하나의 MNIST 데이터에 대해서만 추론을 수행했지만, 이번에는 10개의 MNIST 데이터들을 한 번에 추론한다. 따라서, 10개의 MNIST 데이터들은 연속된 주소에 위치해야 하기 때문에 아래와 같이 host/device memory를 할당한다.
```c++
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
```
여기서 입력과 출력의 차원은 엔진을 쿼리하여 얻을 수 있는데, 이때, 반환되는 차원에서 runtime dimension은 `-1`로 표현된다. 따라서, 쿼리한 차원을 바로 출력해보면 다음과 같이 출력하는 것을 확인할 수 있다.
```
> input  dims: (-1, 1, 28, 28)
> output dims: (-1, 10)
```
우리는 batch size를 10으로 설정하기 때문에 쿼리한 차원의 첫 번째 차원 값을 10으로 설정해주고, 이 크기만큼 메모리를 할당한다.

입력으로 사용할 데이터 또한 10개의 데이터가 연속한 메모리에 위치해야 하므로, 다음과 같이 입력 데이터를 읽고 device memory로 한 번에 복사해주었다.
```c++
// copy mnist data from host to device
for (int i = 0; i < 10; i++) {
    std::string filename = "digits/" + std::to_string(i) + ".bin";
    loadBinary((void*)(input + i * 28 * 28), 28 * 28, filename.c_str());
}
cudaMemcpy(d_input, input, sizeof(float) * in_count, cudaMemcpyHostToDevice);
```

# Performing Inference

추론을 수행할 준비는 거의 다 되었다. 이제 엔진을 통해 execution context를 생성한다. 여기서는 `euqueueV3()`를 사용하여 추론을 수행할 예정이므로, 이에 맞게 바인딩 텐서의 주소를 전달해야 한다. `enqueueV2()`의 경우에는 각 텐서의 바인딩 인덱스 위치에 해당 메모리 주소를 배열로 설정하여 API를 호출할 때 전달했지만, `enqueueV3()`는 호출하기 전에 또 다른 API를 통해서 메모리 주소를 설정한다. 이 API는 `setTensorAddress()`이며, 파라미터로 바인딩 텐서의 이름과 이 텐서의 메모리 주소를 전달한다.
```c++
IExecutionContext* context = engine->createExecutionContext();
context->setTensorAddress(engine->getBindingName(0), d_input);
context->setTensorAddress(engine->getBindingName(1), d_output);
```

바인딩 텐서의 메모리 주소 설정을 모두 마쳤으면, 실제 추론에서의 입력 차원을 설정한다. 현재 execution context는 입력의 크기가 `(-1, 1, 28, 28)`로 인식하고 있다. 따라서, 추론에서의 입력 차원이 `(10, 1, 28, 28)`이라고 설정해주는 작업이 필요하다.
```c++
context->setBindingDimensions(0, input_dims);
```

이렇게 설정이 다 완료되었는지 체크하기 위해서 `allInputShapesSpecified()`를 제공하며, 다음과 같이 사용하여 추론을 수행할 수 있다.
```c++
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
```

추론 결과는 다음의 코드를 통해서 확인할 수 있으며, 추론에 걸린 시간 또한 출력해주고 있다.
```c++
// extract output
cudaMemcpy(output, d_output, sizeof(float) * out_count, cudaMemcpyDeviceToHost);

for (int i = 0; i < 10; i++) {
    auto iter = std::max_element((float*)(output + i * 10), (float*)(output + (i + 1) * 10));
    int output_digit = std::distance((float*)(output + i * 10), iter);
    std::cout << "Digit: " << output_digit << " (" << get_prob((float*)(output + i * 10), output_digit) << ")\n";
}
std::cout << "Elapsed Time: " << msec << " ms\n\n";
```

RTX 3080에서의 결과는 다음과 같다. 입력은 0부터 9까지의 MNIST 데이터를 사용헀기 때문에 실제 정답과 일치하며 3을 제외한 데이터들은 모두 99%의 확률로 정답을 맞추었다.
```
Digit: 0 (0.999866)
Digit: 1 (0.999617)
Digit: 2 (1)
Digit: 3 (0.826708)
Digit: 4 (1)
Digit: 5 (0.999913)
Digit: 6 (1)
Digit: 7 (0.999999)
Digit: 8 (0.999973)
Digit: 9 (0.999949)
Elapsed Time: 0.221056 ms
```

[Mnist CNN using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)에서 하나의 데이터를 추론하는데 걸린 시간이 약 0.03 ms 이었지만, 10개의 데이터를 한 번에 추론한 시간은 약 0.22 ms가 걸렸다. TensorRT 문서 내에서 언급한 것과 같이 batch inference를 수행할 때 조금 더 좋은 성능을 보여준다.

<br>

# References

- [NVIDIA TensorRT Documentation: The C++ API](/tensorrt/doc/01_developer_guide/03_the_cpp_api.md)
- [NVIDIA TensorRT Documentation: Working with Dynamic Shapes](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md)