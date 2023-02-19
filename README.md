NVIDIA CUDA, cuDNN, TensorRT 등 NVIDIA 플랫폼에서 제공되는 라이브러리 공식 문서 분석 및 스터디 

- [References](#references)
- [CUDA](#cuda)
  - [CUDA Documentations](#cuda-documentations)
  - [CUDA Study](#cuda-study)

# References

- NVIDIA 공식 문서
- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher

# CUDA

## CUDA Documentations

- [Programming Guide](/cuda-doc/01_programming_guide/01_intro.md)
  - [Programming Model](/cuda-doc/01_programming_guide/02_programming_model.md)
  - Compilation with NVCC
  - [CUDA Runtime](/cuda-doc/01_programming_guide/03_02_cuda_runtime.md)
    - [Device Memory](/cuda-doc/01_programming_guide/03_03_device_memory.md)
    - Device Memory L2 Access Management


## CUDA Study

- [Heterogeneous Computing](/cuda-study/01_heterogeneous_computing.md)
- [CUDA Programming Model](/cuda-study/02_cuda_programming_model.md) (Example: Vector Addition)
- [Organizing Parallel Threads](/cuda-study/03_organizing_parallel_threads.md) (Example: Matrix Addition)
- [Device Query](/cuda-study/04_device_query.md)
- [CUDA Execution Model](/cuda-study/05_cuda_execution_model.md) (GPU Architecture Overview)
- [Understanding Warp Execution and Warp Divergence](/cuda-study/06_understanding_warp_execution.md)
- [Avoiding Branch Divergence](/cuda-study/07_avoiding_branch_divergence.md) (Example: Sum Reduction)
- [Unrolling Loops](/cuda-study/08_unrolling_loops.md) (Example: Sum Reduction)
- [CUDA Memory Model](/cuda-study/09_cuda_memory_model.md) (CUDA Memory Types Overview)
- [Memory Management](/cuda-study/10_memory_management.md) (Pinned Memory / Zero-copy Memory / UVA / Unified Memory)