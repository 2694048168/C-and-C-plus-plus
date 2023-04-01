cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_75;-code=sm_75")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_executable(cuBLAS)
target_sources(cuBLAS
    PRIVATE
        "src/cuda_blas.cu"
        "src/matrix.cpp"
)
target_link_libraries(cuBLAS cublas)
