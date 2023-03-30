cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# nvcc 先将设备代码编译为 PTX（Parallel Thread eXecution）伪汇编代码
# 再将 PTX 代码编译为二进制的 cubin 目标代码.
# 在将源代码编译为 PTX 代码时, 需要用选项 -arch=compute_XY 
#     指定一个虚拟架构的计算能力, 用以确定代码中能够使用的 CUDA 功能;
# 在将 PTX 代码编译为 cubin 代码时, 需要用选项 -code=sm_ZW
#     指定一个真实架构的计算能力, 用以确定可执行文件能够使用的 GPU;
# 真实架构的计算能力必须等于或者大于虚拟架构的计算能力.

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_75;-code=sm_75")

# 如果希望编译出来的可执行文件能够在更多的 GPU 中执行
# 可以同时指定多组计算能力, 每一组用如下形式的编译选项
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60")

add_executable(multipleDimGrid multipleDim_grid.cu)
