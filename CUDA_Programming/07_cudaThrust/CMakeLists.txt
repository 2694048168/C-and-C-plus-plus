cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_75;-code=sm_75")

add_executable(cudaThrust src/scan_vector.cu)
