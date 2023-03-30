cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_75;-code=sm_75")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_executable(cudaError)
target_sources(cudaError
    PRIVATE
        "src/cudaError.cu"
        "src/add.cu"
)
