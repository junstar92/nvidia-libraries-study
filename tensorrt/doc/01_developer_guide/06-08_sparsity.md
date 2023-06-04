# Table of Contents

- [Table of Contents](#table-of-contents)
- [Sparsity](#sparsity)
- [References](#references)

<br>

# Sparsity

NVIDIA Ampere 아키텍처 GPU에서는 [Structured Sparsity](https://blogs.nvidia.com/blog/2020/05/14/sparsity-ai-inference/)를 지원한다. 이 기능을 사용하면서 고성능의 추론을 달성하기 위해서는 convolution 커널의 weights와 fully connected 레이어의 weights가 반드시 아래의 요구사항을 만족해야 한다.

커널 weights에서 각 output channel과 각 spatial pixel에 대해서 4개의 input channel마다 최소 2개의 0이 있어야 한다. 즉, 커널 weight의 shape가 [`K`, `C`, `R`, `S`]이고 `C % 4 == 0`이라고 한다면, 아래의 알고리즘으로 요구사항을 검증할 수 있다.
```python
hasSparseWeights = True
for k in range(0, K):
    for r in range(0, R):
        for s in range(0, S):
            for c_packed in range(0, C // 4):
                if numpy.count_nonzero(weights[k, c_packed*4:(c_packed+1)*4, r, s]) > 2:
                    hasSparseWeights = False
```

Sparsity feature를 활성화하려면, builder config를 통해 `kSPARSE_WEIGHTS`를 설정하고 `kFP16` 또는 `kINT8` 모드가 활성화되어야 한다. 예를 들면, 다음과 같이 지정해야 한다.
```c++
config->setFlag(BuilderFlag::kSPARSE_WEIGHTS);
config->setFlag(BuilderFlag::kFP16);
config->setFlag(BuilderFlag::kINT8);
```

TensorRT 엔진을 빌드할 때, 로그의 끝에 보면 어떤 레이어가 structured sparsity 요구사항을 만족하는지와 어떤 레이어가 structured sparsity를 사용하는 tactics를 선택했는지 알 수 있다. 어떤 경우에서는 structured sparsity를 사용하는 tactics이 normal tactics보다 더 느릴 수 있고, 이 경우에는 TensorRT는 normal tactics을 선택한다. 예를 들면, 아래와 같이 출력된다.
```
[03/23/2021-00:14:05] [I] [TRT] (Sparsity) Layers eligible for sparse math: conv1, conv2, conv3
[03/23/2021-00:14:05] [I] [TRT] (Sparsity) TRT inference plan picked sparse implementation for layers: conv2, conv3
```

강제로 kernel weights가 structured sparsity patterns이 되도록 하면 정확도 손실이 발생할 수 있다. Fine-tuning을 사용하여 손실된 정확도를 회복시키려면, [Automatic SParsity tool in PyTorch](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) 내용을 통해 살펴볼 수 있다.

> `trtexec`를 사용하여 structured sparsity의 추론 성능을 측정하는 방법은 [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) 섹션을 참조 바람.

<br>

# References

- [NVIDIA TensorRT Documentation: Sparsity](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#structured-sparsity)
- [How Sparsity Adds Umph to AI Inference](https://blogs.nvidia.com/blog/2020/05/14/sparsity-ai-inference/)
- [Automatic SParsity tool in PyTorch](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)