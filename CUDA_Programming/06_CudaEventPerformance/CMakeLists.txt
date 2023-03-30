cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_75;-code=sm_75")

# 添加宏变量编译选项
add_definitions(-DUSE_DoublePrecision)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_executable(cudaPerformance)
target_sources(cudaPerformance
    PRIVATE
        "src/main.cu"
        "src/add.cu"
)
if(CUDA_FOUND)
    target_link_libraries(cudaPerformance ${CUDA_LIBRARIES})
endif()
