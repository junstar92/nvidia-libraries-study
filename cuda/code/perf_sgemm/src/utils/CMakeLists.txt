set(UTILS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(UTILS_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(utils)
set_target_properties(utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(utils
    PUBLIC
        ${UTILS_INC_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${UTILS_SRC_DIR}
)
target_sources(utils
PRIVATE
    ${UTILS_SRC_DIR}/cuda_utils.cpp
)
set_source_files_properties(${UTILS_SRC_DIR}/cuda_utils.cpp
PROPERTIES
    LANGUAGE CUDA
)