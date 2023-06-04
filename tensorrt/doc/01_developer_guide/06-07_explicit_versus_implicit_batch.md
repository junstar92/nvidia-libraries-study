# Table of Contents

- [Table of Contents](#table-of-contents)
- [Explicit Versus Implicit Batch](#explicit-versus-implicit-batch)
- [References](#references)

<br>

# Explicit Versus Implicit Batch

TensorRT는 네트워크를 지정할 때 아래의 두 가지 모드를 지원한다.

- Explicit Batch
- Implicit Batch

_Implicit Batch_ mode에서 모든 텐서는 implicit batch dimension을 가지고 다른 차원들은 반드시 상수 길이를 갖는다. 이 모드는 TensorRT 초기 버전에서 사용되었고, 이제는 deprecated 되었으며 backward compatibility를 위해 지원된다.

_Explicit Batch_ mode에서 모든 차원은 explicit이며 dynamic할 수도 있다. 즉, 그 길이는 실행 시간에 변경될 수 있다. Dynamic shapes와 loops와 같은 새로운 많은 기능들은 거의 대부분 explicit batch mode에서만 이용 가능하다. ONNX parser에서도 explicit batch mode가 필요하다.

예를 들어, 사이즈가 HxW인 3채널 이미지 N개를 처리한다고 가정해보자 (`NCHW` format). 런타임에서 입력 텐서의 차원은 [N,3,H,W]이다. 두 모드에서 `INetworkDefinition`은 이 입력 텐서의 차원을 아래와 같이 처리한다.

- In explicit batch mode, the network specifies [N,3,H,W].
- In implicit batch mode, the network specifies only [3,H,W]. The batch dimension `N` is implicit.

네트워크에서 batch dimension을 지정할 방법이 없기 때문에 implicit batch mode에서는 배치 간 연산을 표현할 방법이 없다. 예를 들어, implicit batch mode에서 다음과 같은 연산은 불가능하다.

- reducing across the batch dimension
- reshaping the batch dimension
- transposing the batch dimension with another dimension

한 가지 예외는 네트워크 입력에 대해 `ITensor::setBroadcastAcrossBatch` 메소드와 다른 텐서의 implicit 브로드캐스팅을 통해 배치 전체에 텐서를 브로드캐스트할 수 있다는 것이다.

Explicit batch 모드에서는 제한을 없앤다. 배치 축은 단지 첫 번째 축(axis 0)이다. Explicit batch에 대한 더 정확한 용어는 "batch oblivious"이다. TensorRT는 특정 연산을 제외하고는 더 이상 선행 차원에 특별한 의미를 부여하지 않는다. 실제로 explicit batch mode에서는 배치 차원이 없거나(하나의 이미지만 처리하는 네트워크), 관련없는 것들의 배치 차원이 될 수 있다.

Explicit과 implicit batch mode는 `INetworkDefinition`을 생성할 때 플래그를 사용하여 선택할 수 있다. C++에서는 아래와 같이 작성하여 explicit batch mode를 선택할 수 있다.
```c++
IBuilder* builder = ...;
INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
```

Implicit batch 모드를 선택하려면 `createNetwork`를 사용하거나 `createNetworkV2`에 0을 인자로 전달하면 된다.

<br>

# References

- [NVIDIA TensorRT Documentation: Explicit Versus Implicit Batch](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)