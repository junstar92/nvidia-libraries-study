set(RUNNER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(RUNNER_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(runner)
set_target_properties(runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(runner
    PUBLIC ${RUNNER_INC_DIR}
    PRIVATE ${RUNNER_SRC_DIR}
)
target_sources(runner
PRIVATE
    ${RUNNER_SRC_DIR}/runner.cpp
    ${RUNNER_SRC_DIR}/timer.cpp
)
set_source_files_properties(
    ${RUNNER_SRC_DIR}/runner.cpp
    ${RUNNER_SRC_DIR}/timer.cpp
PROPERTIES
    LANGUAGE CUDA
)
target_link_libraries(runner
PRIVATE
    utils kernel
)