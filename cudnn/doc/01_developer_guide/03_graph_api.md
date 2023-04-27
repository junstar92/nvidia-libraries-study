# Table of Contents
- [Table of Contents](#table-of-contents)
- [Graph API](#graph-api)
- [Key Concepts](#key-concepts)
  - [Operations and Operation Graphs](#operations-and-operation-graphs)
  - [Engines and Engine Configurations](#engines-and-engine-configurations)
  - [Heuristics](#heuristics)
- [Graph API Example with Operation Fusion](#graph-api-example-with-operation-fusion)
  - [Creating Operation and Tensor Descriptors to Specify the Graph Dataflow](#creating-operation-and-tensor-descriptors-to-specify-the-graph-dataflow)
  - [Finalizing The Operation Graph](#finalizing-the-operation-graph)
  - [Configuring An Engine That Can Execute The Operation Graph](#configuring-an-engine-that-can-execute-the-operation-graph)
  - [Executing The Engine](#executing-the-engine)
- [Supported Graph Patterns](#supported-graph-patterns)
  - [Pre-compiled Single Operation Engines](#pre-compiled-single-operation-engines)
  - [Generic Runtime Fusion Engines](#generic-runtime-fusion-engines)
  - [Specialized Runtime Fusion Engines](#specialized-runtime-fusion-engines)
  - [Specialized Pre-Compiled Engines](#specialized-pre-compiled-engines)
  - [Mapping with Backend Descriptors](#mapping-with-backend-descriptors)
- [References](#references)

<br>

# Graph API

cuDNN 라이브러리는 연산 그래프로 계산을 설명하는 선언적 프로그래밍(declarative programming) 모델을 제공한다. 연산 퓨전(computation fusion)의 중요성이 증가함에 따라 더 유연한 API를 제공하기 위해 graph API가 cuDNN 8.0에서 도입되었다.

사용자는 연산 그래프를 작성하는 것으로 시작한다. 고수준에서 사용자는 텐서 연산에서 데이터 흐름의 그래프를 설명한다. 최종 그래프가 주어지면 사용자는 해당 그래프를 실행할 수 있는 엔진을 선택하고 구성한다. 사용 편의성, 런타임 오버헤드, 엔진의 성능과 관련하여 장단점이 있는 엔진을 선택하고 구성하는 방법에는 여러 가지가 있다.

Graph API에는 두 가지 진입 포인트가 있다.

- [cuDNN Backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api) (lowest level entry point into the graph API)
- [cuDNN Frontend API](https://github.com/NVIDIA/cudnn-frontend) (convenience layer on top of the C backend API)

대부분 유저에게는 cuDNN frontend API가 더 선호될 것이라고 추측한다. Frontend API의 특징은 다음과 같다.

- 덜 장황하다. Backend API를 통해 액세스할 수 있는 기능들은 frontend API를 통해서도 액세스할 수 있으며, 더 직관적이다.
- Backend API의 위에 errata filter나 autotuning 등과 같은 기능을 추가한다.
- 오픈 소스이다.

<br>

# Key Concepts

Graph API의 주요 컨셉은 다음과 같다.

- [Table of Contents](#table-of-contents)
- [Graph API](#graph-api)
- [Key Concepts](#key-concepts)
  - [Operations and Operation Graphs](#operations-and-operation-graphs)
  - [Engines and Engine Configurations](#engines-and-engine-configurations)
  - [Heuristics](#heuristics)
- [Graph API Example with Operation Fusion](#graph-api-example-with-operation-fusion)
  - [Creating Operation and Tensor Descriptors to Specify the Graph Dataflow](#creating-operation-and-tensor-descriptors-to-specify-the-graph-dataflow)
  - [Finalizing The Operation Graph](#finalizing-the-operation-graph)
  - [Configuring An Engine That Can Execute The Operation Graph](#configuring-an-engine-that-can-execute-the-operation-graph)
  - [Executing The Engine](#executing-the-engine)
- [Supported Graph Patterns](#supported-graph-patterns)
  - [Pre-compiled Single Operation Engines](#pre-compiled-single-operation-engines)
  - [Generic Runtime Fusion Engines](#generic-runtime-fusion-engines)
  - [Specialized Runtime Fusion Engines](#specialized-runtime-fusion-engines)
  - [Specialized Pre-Compiled Engines](#specialized-pre-compiled-engines)
  - [Mapping with Backend Descriptors](#mapping-with-backend-descriptors)
- [References](#references)

## Operations and Operation Graphs

연산 그래프(operation graph)는 텐서에 대한 연산의 dataflow graph이다. 이는 수학적 사양(mathematical specifications)를 의미하며, 그래프에서 사용할 수 있는 엔진이 둘 이상일 수 있으므로 그래프는 이를 실행할 수 있는 엔진과 분리되어 있다.

I/O 텐서는 암시적으로 연산들을 연결한다. 예를 들어, 연산 A가 텐서 X를 생성하고 이 텐서 X가 연산 B에서 사용된다면, 연산 B가 연산 A에 종속된다는 것을 의미한다.

## Engines and Engine Configurations

주어진 연산 그래프를 실행할 수 있는 몇 가지 엔진 후보군이 있다. 이러한 엔진 목록을 쿼리하는 일반적인 방법은 아래에서 설명할 휴리스틱 쿼리(heuristics query)를 사용하는 것이다.

각 엔진에는 엔진의 속성을 구성할 수 있는 노브(knobs, refer to [`cudnnBackendKnobType_t'](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBackendKnobType_t))들이 있다.

## Heuristics

휴리스틱은 주어진 연산 그래프에 대해 가장 성능이 좋은 것부터 가장 낮은 성능 순으로 정렬된 엔진 구성 리스트를 가져오는 방법이다. 여기에는 3가지 모드가 있다.

- `CUDNN_HEUR_MODE_A` - 가능한 대부분의 연산 그래프 패턴을 처리할 수 있고 빠르게 수행하도록 의도되어 있다. 예상 성능에 따라서 엔진 구성 리스트를 반환한다.
- `CUDNN_HEUR_MODE_B` - 일반적으로 A 모드보다 더 정확하게 동작하도록 의도되었지만, CPU의 latency가 높아질 수 있다. 예상 성능에 따라 엔진 구성 리스트를 반환한다. 만약 A 모드가 더 좋은 결과를 반환하는 경우에는 모드 A 휴리스틱으로 fallback될 수 있다.
- `CUDNN_HEUR_MODE_FALLBACK` - 빠르면서 최적의 성능을 기대하지 않는 fallback을 제공하기 위한 것이다.

권장하는 방법은 모드 A 또는 모드 B를 쿼리하고 지원 여부를 확인하는 것이다. 지원되는 엔진 중 첫 번째 엔진 구성이 가장 좋은 성능을 보여줄 것이라고 예상된다.

특정 device에서 특정 문제에 대해 가장 좋은 엔진을 선택하려면, 각 엔진마다 실행 시간을 측정하고 최적의 엔진을 선택하는 "auto-tune"을 사용할 수 있다. cuDNN frontend API에서는 이를 위해 `cudnnFindPlan()`이라는 함수를 제공한다.

만약 지원되는 엔진 구성이 없다면 functionall fallbacks를 찾기 위해 fallback 모드를 사용한다.

전문적으로 사용하기 위해 엔진 속성을 기반으로 엔진 구성을 필터링할 수 있다. 이러한 속성에는 **numerical notes**, **behavior notes**, 또는, **adjustable knobs**가 있다. Nueraical notes는 입력이나 출력에서 datatype down conversion이 일어나는지와 같은 numerical 속성을 유저에게 제공한다. Behavior notes는 런타임 컴파일을 사용하는지 여부와 같은 백엔드 구현에 대한 정보를 나타낼 수 있다. Adjustable knobs를 사용하면 엔진의 동작 및 성능을 세밀하게 제어할 수 있다.

<br>

# Graph API Example with Operation Fusion

## Creating Operation and Tensor Descriptors to Specify the Graph Dataflow

먼저, 3개의 cuDNN backend operation descriptors를 생성한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcAJGMX%2FbtsbUqgcY9Y%2FkGbskzbxp5bVKpPgM7ks4K%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

위 그림에서와 같이, 유저는 하나의 forward convolution operation (using `CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`), bias addition을 위한 하나의 pointwise operation (using `CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR` with `CUDNN_POINTWISE_ADD`), 그리고 ReLU activation을 위한 하나의 pointwise operation (using `CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR` with mode `CUDNN_POINTWISE_RELU_FWD`)를 지정할 수 있다. 디스크립터들의 속성을 설정하는 방법은 API 문서에서 자세히 설명하고 있다. cuDNN backend API 문서의 [Setting Up An Operation Graph For A Grouped Convolution use case](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#use-case-op-graph-group-convo)에서 forward convolution을 설정하는 예제를 보여준다.

또한, 그래프의 모든 연산의 입력과 출력에 대한 tensor descriptors를 생성해주어야 한다. 그래프의 데이터 흐름은 텐서 할당으로 추론된다 (위 그림 참조). 예를 들어, 백엔드 텐서 `Tmp0`을 convolution 연산의 출력 및 bias 연산의 입력으로 지정함으로써, cuDNN은 데이터 흐름이 convolution에서 bias로 실행된다는 것을 추론한다. `Tmp1` 텐서에도 동일한 원리가 적용된다. 사용자가 중간 결과인 `Tmp0`과 `Tmp1`을 다른 용도로 사용하지 않는 경우에는 virtual 텐서로 지정하여 메모리 I/O를 최적화할 수 있다.

하나 이상의 연산 노드를 가진 그래프는 in-place 연산(입력 UID 중 어느 하나가 출력 UID들 중 하나와 일치하는 경우)을 지원하지 않는다. 이러한 in-place 연산은 후속 그래프 분석(graph analysis)에서 순환으로 간주되어 지원되지 않는다. in-place 연산은 오직 단일 노드 그래프에서만 지원된다. 또한, operation descriptors는 텐서의 UIDs만으로도 그래프의 의존성을 결정할 수 있기 때문에, 어떤 순서로 생성하고 cuDNN에 전달해도 상관없다.

## Finalizing The Operation Graph

두 번째로, 사용자가 연산 그래프를 마무리한다. 이 과정에서 cuDNN 연산 간의 종속 관계를 설정하고 엣지(edge)를 연결하기 위한 dataflow 분석을 수행한다. 이 단계에서 cuDNN은 그래프의 유효성을 확인하기 위해 다양한 검사를 수행한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbsJBDU%2Fbtscjj0Q9Qq%2FbnYcSXIOqVopwue8mSAtT0%2Fimg.png" height=300px style="display: block; margin: 0 auto; background-color:white"/>

## Configuring An Engine That Can Execute The Operation Graph

세번째, 완성된 연산 그래프가 주어지면, 해당 그래프를 실행할 엔진을 선택하고 구성하며, 이는 execution plan을 생성한다. [Heuristics](#heuristics)에서 언급한 대로, 일반적인 방법은 다음과 같다.

1. 휴리스틱 모드 A 또는 B를 쿼리한다.
2. 기능적으로 지원하는 첫 번째 엔진 구성을 선택한다 (또는, 기능적으로 지원하는 모든 엔진 구성을 auto-tune 한다).
3. 2번 과정에서 엔진 구성을 찾을 수 없는 경우, 더 많은 옵션을 위해 fallback 휴리스틱을 쿼리한다.

## Executing The Engine

마지막으로, execution plan이 생성되고 실행되어야 할 때, 사용자는 workspace 포인터, UID 배열 및 device pointer 배열을 제공하여 backend variant pack을 구성해야 한다. UIDs와 포인터는 대응하는 순서대로 배열되어야 한다. Handle, execution plan, variant pack이 준비되면, execution API를 호출하여 연산이 GPU에서 수행된다.

<br>

# Supported Graph Patterns

cuDNN Graph API는 일련의 그래프 패턴을 지원한다. 이러한 패턴들은 각각 자체 `support surfaces(?)`를 가진 많은 엔진들에 의해 지원된다. 이러한 엔진들은 아래 4개의 클래스로 그룹화된다.

- pre-compiled single operation engines
- generic runtime fusion engines
- specialized runtime fusion engines
- specialized pre-compiled fusion engines

Runtime compilation 또는 pre-compilation을 사용하는 specialized engines는 중요한 use-case를 타겟으로 하며 현재 지원하는 패턴이 꽤 제한적이다. 시간이 자니면서 더 많은 use-cases를 generic runtime fusion engines로 지원할 예정이라고 한다.

이러한 엔진들이 지원하는 패턴은 겹치는 부분도 있기 때문에, 주어진 패턴에서 0개, 1개, 또는 여러 개의 엔진을 가져올 수 있다.

아래에서는 각 엔진 클래스에 대해 간략하게 설명하며, 어떤 그래프 패턴들을 지원하는지는 문서를 참조 바람([link](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#support-graph-patterns)).

## Pre-compiled Single Operation Engines

Pre-compiled engines는 하나의 연산 그래프만을 지원하며, `ConvolutionFwd`, `ConvolutionBwFilter`, `ConvolutionBwData`, `ConvolutionBwBias` 등이 있다.

## Generic Runtime Fusion Engines

Pre-compiled single operation engines는 단일 연산 패턴을 지원한다. 당연히 퓨전이 쓸모있으려면 그래프가 여러 연산을 지원해야 하며, 이상적으로는 지원되는 패턴이 다양한 use-cases를 커버하기 위해 유연해야 한다. 이러한 일반성(generality)을 달성하기 위해 cuDNN은 그래프 패턴 기반으로 런타임에 커널들을 생성하는 runtime fusion engines를 가지고 있다. 이 섹션에는 이러한 runtime fusion engines (즉, `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION` behavior note를 갖는 엔진)에서 지원되는 패턴을 다루고 있다.

## Specialized Runtime Fusion Engines

Specialized runteim fusion engeins는 인기가 많은 딥러닝 모델에서 일반적으로 발생하는 특수한 그래프 패턴을 대상으로 최적화한다. 이러한 엔진은 지원되는 fusion patterns, data types, tensor layouts에 따라 제한된 유연성을 제공한다. 장기적인 관점에서 이러한 패턴들은 더 일반적이게 될 것이라 예상한다고 한다.

## Specialized Pre-Compiled Engines

Pre-compiled specialized engines는 ragged support surface를 가진 specialized graph pattern을 타겟으로 최적화하며, 이러한 그래프들은 런타임 컴파일이 필요하지 않는다.

대부분의 경우, specialized patterns은 runtime fusion engines에서 사용되는 일반적인 패턴의 특수한 경우일 뿐이지만, 이러한 패턴이 일반적인 패턴과 맞지 않는 경우도 있다. 만약 그래프 패턴이 specialized pattern과 일치한다면, 적어도 패턴이 일치하는 엔진을 얻을 수 있으며 다른 옵션으로 runtime fusion 엔진을 얻을 수도 있다.

## Mapping with Backend Descriptors

문서에서는 각 그래프 패턴을 설명할 때, 연산을 익숙하면서도 읽기 쉬운 표기로 표현하고 있다. 각 연산들과 매핑되는 백엔드 descriptors를 [table](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#mapping-backend-desc)에서 보여주고 있다. 표에서 언급하는 descriptor외에도 다른 것들이 있으므로 전체 내용은 API 문서를 참조해야 한다.



<br>

# References
- [NVIDIA cuDNN Documentation: Graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)