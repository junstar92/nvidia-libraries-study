set(KERNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(KERNEL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(kernel)
target_include_directories(kernel
    PUBLIC ${KERNEL_INC_DIR} ${KERNEL_SRC_DIR}
)
target_sources(kernel
PRIVATE
    ${KERNEL_SRC_DIR}/00_cublas_gemm.cu
    ${KERNEL_SRC_DIR}/01_naive_gemm.cu
    ${KERNEL_SRC_DIR}/02_smem_gemm.cu
    ${KERNEL_SRC_DIR}/03_1d_blocktiling_gemm.cu
    ${KERNEL_SRC_DIR}/04_2d_blocktiling_gemm.cu
    ${KERNEL_SRC_DIR}/05_vectorize_gemm.cu
    ${KERNEL_SRC_DIR}/06_warptiling_gemm.cu
)

target_compile_options(kernel PRIVATE "--ptxas-options=-v -O2")
target_link_libraries(kernel
PUBLIC
    CUDA::cublas
)