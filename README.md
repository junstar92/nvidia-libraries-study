NVIDIA CUDA, cuDNN, TensorRT 등 NVIDIA 플랫폼에서 제공되는 라이브러리 공식 문서 분석 및 스터디 

- [References](#references)
- [CUDA](#cuda)
  - [CUDA Documentations](#cuda-documentations)
    - [Programming Guide](#programming-guide)
  - [CUDA Study](#cuda-study)

# References

- NVIDIA 공식 문서
- Professional CUDA C Programming By John Cheng, Max Grossman, Ty Mckercher

# CUDA

## CUDA Documentations

### Programming Guide

- [Programming Guide Intro](/cuda/doc/01_programming_guide/01_intro.md)
- [Programming Model](/cuda/doc/01_programming_guide/02_programming_model.md)
- Compilation with NVCC
- [CUDA Runtime](/cuda/doc/01_programming_guide/03-02_cuda_runtime.md)
  - [Device Memory](/cuda/doc/01_programming_guide/03-03_device_memory.md)
  - Device Memory L2 Access Management


## CUDA Study

- [Heterogeneous Computing](/cuda/study/01_heterogeneous_computing.md)
- [CUDA Programming Model](/cuda/study/02_cuda_programming_model.md) (Example: Vector Addition)
- [Organizing Parallel Threads](/cuda/study/03_organizing_parallel_threads.md) (Example: Matrix Addition)
- [Device Query](/cuda/study/04_device_query.md)
- [CUDA Execution Model](/cuda/study/05_cuda_execution_model.md) (GPU Architecture Overview)
- [Understanding Warp Execution and Warp Divergence](/cuda/study/06_understanding_warp_execution.md)
- [Avoiding Branch Divergence](/cuda/study/07_avoiding_branch_divergence.md) (Example: Sum Reduction)
- [Unrolling Loops](/cuda/study/08_unrolling_loops.md) (Example: Sum Reduction)
- [CUDA Memory Model](/cuda/study/09_cuda_memory_model.md) (CUDA Memory Types Overview)
- [Memory Management](/cuda/study/10_memory_management.md) (Pinned Memory / Zero-copy Memory / UVA / Unified Memory)
  -  [Example: Matrix Addition with Unified Memory](/cuda/study/10-1_matrix_addition_with_unified_memory.md)
- [Global Memory Access Patterns](/cuda/study/11_memory_access_patterns.md)
  - [Example: Matrix Transpose](/cuda/study/11-1_matrix_transpose_problem.md)
- [Introducing CUDA Shared Memory](/cuda/study/12_shared_memory.md) (Shared Memory Bank / Synchronization / Volatile Qualifier)
  - [Layout of Shared Memory](/cuda/study/12-1_data_layout_of_shared_memory.md) (Square & Rectangular Shared Memory)
  - [Reducing Global Memory Access](/cuda/study/12-2_reducing_global_memory_access.md) (Example: Sum Reduction with Shared Memory)
  - [Coalescing Global Memory Accesses](/cuda/study/12-3_coalescing_global_memory_accesses.md) (Example: Matrix Transpose with Shared Memory)
- [Constant Memory and Read-Only Cache](/cuda/study/13_constant_memory.md)
- [Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md)