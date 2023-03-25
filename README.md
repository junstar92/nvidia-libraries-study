CUDA, cuDNN, TensorRT 등 NVIDIA 라이브러리 공식 문서 분석 및 스터디 

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

> CUDA 12.1 문서를 기준으로 작성됨

### Programming Guide

- [Programming Guide Intro](/cuda/doc/01_programming_guide/01_intro.md)
- [Programming Model](/cuda/doc/01_programming_guide/02_programming_model.md)
- Programming Interface
  - Compilation with NVCC
  - [CUDA Runtime](/cuda/doc/01_programming_guide/03-02_cuda_runtime.md)
    - [Device Memory](/cuda/doc/01_programming_guide/03-02-02_device_memory.md)
    - Device Memory L2 Access Management
    - [Shared Memory](/cuda/doc/01_programming_guide/03-02-04_shared_memory.md) (Example: Matrix Multiplication with Shared Memory)
    - [Page-Locked Host Memory](/cuda/doc/01_programming_guide/03-02-06_page_locked_host_memory.md)
    - [Asynchronous Concurrent Execution](/cuda/doc/01_programming_guide/03-02-08_asynchronous_concurrent_execution.md) (CUDA Streams)
      - Programmatic Dependent Launch and Synchronization
      - CUDA Graphs
      - [CUDA Events](/cuda/doc/01_programming_guide/03-02-08-08_cuda_events.md)
      - Synchronous Calls
    - Multi-Device System
    - [Unified Virtual Address Space](/cuda/doc/01_programming_guide/03-02-10_unified_virtual_address_space.md)
  - [Versioning and Compatibility](/cuda/doc/01_programming_guide/03-03_versioning_and_compatibility.md) (호환성 관련 내용)
- Hardware Implementation
- Performance Guidelines
  - [Maximize Utilization](/cuda/doc/01_programming_guide/03-05-02_maximize_utilization.md)
  - [Maximize Memory Throughput](/cuda/doc/01_programming_guide/03-05-03_maximize_memory_throughput.md)
  - Maximize Instruction Throughput
  - Minimize Memory Thrashing
- C++ Language Extensions


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
- [Introducing CUDA Streams](/cuda/study/14_introducing_cuda_streams.md) (+ False Dependency, Hyper-Q)
  - [Concurrent Kernel Execution](/cuda/study/14-1_concurrent_kernel_execution.md)
  - [Overlapping Kernel Execution and Data Transfer](/cuda/study/14-2_overlapping_kernel_execution_and_data_transfer.md)
  - [Stream Callback](/cuda/study/14-3_stream_callback.md)
- [Introducing CUDA Events](/cuda/study/15_introducing_cuda_event.md)
- [Warp Shuffle Instruction](/cuda/study/16_warp_shuffle.md) (Example: Sum Reduction with Warp Shuffle Instruction)
- [Precision Issues of Floating-Point Number in CUDA](/cuda/study/17_precision_issues_in_cuda.md)