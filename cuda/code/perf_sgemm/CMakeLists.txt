cmake_minimum_required(VERSION 3.17)
project(perf_sgemm LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# check if CUDA is available
find_package(CUDAToolkit)
include(CheckLanguage)
check_language(CUDA)

# set compute capability
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# compiler option
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")

# set build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# subdirectory for adding libraries
add_subdirectory(src)

# add executable for src/main.cpp
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
PRIVATE 
    ${SRC_DIR}/main.cpp
)
target_link_libraries(${PROJECT_NAME}
PRIVATE
    runner
)