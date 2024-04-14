# TensorRT

- [Quick Start Guide](/tensorrt/doc/00_getting_started/01_quick_start_guide.md)

## References

- [NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt)
  - [Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html?ncid=em-prod-790406) (for platform and software version compatibility)
  - [Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html?ncid=em-prod-790406)
  - [Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html?ncid=em-prod-790406)
  - [Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
- [Code Samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
- [Introductory Webinar](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/?ncid=em-prod-790406) : Learn more about TensorRT features and tools that simplify the inference workflow
- ONNX-TensorRT
  - [Github: pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
  - [pytorch-quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html)
  - [PTQ, QAT Workflow](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/quantization_tutorial/qat-ptq-workflow.ipynb)

## Developer Guide

> v8.6.1 Documentation 기준으로 작성됨

- [Introduction](/tensorrt/doc/01_developer_guide/01_introduction.md)
- [TensorRT's Capabilities](/tensorrt/doc/01_developer_guide/02_tensorrts_capabilities.md)
- [The C++ API](/tensorrt/doc/01_developer_guide/03_the_cpp_api.md)
- [How TensorRT Works](/tensorrt/doc/01_developer_guide/05_how_tensorrt_works.md)
- Advanced Topics
  - [About Compatibility](/tensorrt/doc/01_developer_guide/06-01_about_compatibility.md) (Version Compatibility, Hardware Compatibility, Compatibility Checks)
  - Refitting an Engine
  - [Algorithm Selection and Reproducible Builds](/tensorrt/doc/01_developer_guide/06-03_algorithm_selection_and_reproducible_builds.md)
  - [Creating a Network Definition from Scratch](/tensorrt/doc/01_developer_guide/06-04_creating_a_network_definition_from_scratch.md)
  - [Reduced Precision](/tensorrt/doc/01_developer_guide/06-05_reduced_precision.md)
  - [I/O Formats](/tensorrt/doc/01_developer_guide/06-06_io_formats.md)
  - [Explicit Versus Implicit Batch](/tensorrt/doc/01_developer_guide/06-07_explicit_versus_implicit_batch.md)
  - [Sparsity](/tensorrt/doc/01_developer_guide/06-08_sparsity.md)
  - [Empty Tensors](/tensorrt/doc/01_developer_guide/06-09_empty_tensors.md)
  - Reusing Input Buffers
  - [Engine Inspector](/tensorrt/doc/01_developer_guide/06-11_engine_inspector.md)
  - Preview Features
- Working with INT8
  - [Introduction to Quantization](/tensorrt/doc/01_developer_guide/07-01_introducing_to_quantization.md)
  - [Setting Dynamic Range](/tensorrt/doc/01_developer_guide/07-02_setting_dynamic_range.md)
  - [Post-Training Quantization Using Calibration](/tensorrt/doc/01_developer_guide/07-03_post_training_quantization_using_calibration.md)
  - [Explicit Quantization](/tensorrt/doc/01_developer_guide/07-04_explicit_quantization.md)
  - [INT8 Rounding Modes](/tensorrt/doc/01_developer_guide/07-05_int8_rounding_modes.md)
- [Working with Dynamic Shapes](/tensorrt/doc/01_developer_guide/08_working_with_dynamic_shapes.md)
- [Extending TensorRT with Custom Layers](/tensorrt/doc/01_developer_guide/09_extending_tensorrt_with_custom_layers.md)
- Working with Loops
- Working with Conditions
- Working with DLA
- Performance Best Practices
  - [Measuring Performances](/tensorrt/doc/01_developer_guide/13-01_measuring_performance.md)
  - Hardware/Software Environment for Performance Measurements
  - [Optimizing TensorRT Performance](/tensorrt/doc/01_developer_guide/13-03_optimizing_tensorrt_performance.md)
  - [Optimizing Layer Performance](/tensorrt/doc/01_developer_guide/13-04_optimizing_layer_performance.md)
  - [Optimizing for Tensor Cores](/tensorrt/doc/01_developer_guide/13-05_optimizing_for_tensor_cores.md)
  - [Optimizing Plugins](/tensorrt/doc/01_developer_guide/13-06_optimizing_plugins.md)
  - Optimizing Python Performance
  - [Improving Model Accuracy](/tensorrt/doc/01_developer_guide/13-08_improving_model_accuracy.md)
  - [Optimizing Builder Performance](/tensorrt/doc/01_developer_guide/13-09_optimizing_builder_performance.md)
  - [Builder Optimization Level](/tensorrt/doc/01_developer_guide/13-10_builder_optimization_level.md)
- Troubleshooting
  - [Understanding Formats Printed in Logs](/tensorrt/doc/01_developer_guide/13-10_builder_optimization_level.md)

## Study

- [Mnist CNN using TensorRT Network Definition APIs](/tensorrt/study/01_mnist_cnn_api.md)
- [Mnist CNN using ONNX Parser APIs](/tensorrt/study/02_mnist_cnn_onnx.md)
- [Mnist CNN with Dynamic Shapes](/tensorrt/study/03_mnist_cnn_with_dynamic_shapes.md)
- [TensorRT 8.5 업데이트 정리](/tensorrt/study/04_tensorrt_8_5_update.md)
- [TensorRT 8.6 업데이트 정리](/tensorrt/study/05_tensorrt_8_6_update.md)
- [TensorRT 9.X 업데이트 정리](/tensorrt/study/07_tensorrt_9_x_update.md)
- [TensorRT 10.X 업데이트 정리](/tensorrt/study/08_tensorrt_10_x_update.md)
- [Plugin Example 구현](/tensorrt/study/06_plugin.md)