# cuDNN

## References

- [NVIDIA cuDNN Documentation (latest)](https://docs.nvidia.com/deeplearning/cudnn/index.html)
  - [Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
  - [Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
  - [Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)
  - [API Reference](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)
- [NVIDIA cuDNN Documentation (v7.6.5)](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/index.html)
- [GTC 2020: cuDNN v8 New Advances](https://developer.nvidia.com/gtc/2020/video/s21685)
- [cudnn-frontend (github)](https://github.com/NVIDIA/cudnn-frontend) : C++ header-only library that wraps the cuDNN [C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api)

## Developer Guide

> v8.9.0 Documentations 기준으로 작성됨

- [Overview and Core Concepts](/cudnn/doc/01_developer_guide/01_02_overview_and_core_concepts.md)
- [Graph API](/cudnn/doc/01_developer_guide/03_graph_api.md)
- [Legacy API](/cudnn/doc/01_developer_guide/04_legacy_api.md)
- [Odds and Ends](/cudnn/doc/01_developer_guide/05_odds_end_ends.md)

## Study

- [Example: Logistic Regression](/cudnn/study/01_logistic_regression.md)
- [Example: Mnist CNN (using legacy APIs)](/cudnn/study/02_mnist_cnn(v7).md)