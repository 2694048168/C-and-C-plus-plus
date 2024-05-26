# http://cppdebug.com/archives/550
# https://tanyaodan.com/cpp%E7%94%A8vs%E7%BC%96%E8%AF%91opencv/
# https://blog.csdn.net/qq_26157437/article/details/129834852
# https://blog.csdn.net/weixin_43788327/article/details/125823507
# https://blog.csdn.net/FlyToClouds/article/details/136977855
# https://blog.csdn.net/YuZhuQue/article/details/124837437
# https://blog.csdn.net/baobei0112/article/details/123598794

# 设置 CMake 所需要的最小版本
cmake_minimum_required(VERSION 3.20.0)

# set(CUDATOOLKITDIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
if(UNIX)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")
endif()

# 根据不同的硬件架构, 生成不同的代码
# 支持多个显卡意味着要生成多分代码,所以在开发时建议针对硬件平台设置,否则会大大的影响编译的时间
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_ARCHITECTURES 75;86)
# endif()

# 配置项目名称(solution in VS)以及简要信息
# 有两种方法可以启用 CUDA(若 CUDA 的支持不可选)
project(CudaLearningCourse
    VERSION 0.0.1
    DESCRIPTION "The Modern C++ and CUDA fast Course"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX CUDA
)

# MinGW+Clang | Ninja | CMake | VSCode + clangd + c/c++ Snippets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 CUDA 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(Include_Dir ${CMAKE_SOURCE_DIR}/src/)

# 编译选项支持是否开启cuda支持(即cuda为可选项)
# if(CUDA_ENABLE)
# enable_language(CUDA)
# endif()

# 要检查 CUDA 是否可用, 可使用 CheckLanuage:
include(CheckLanguage)
check_language(CUDA)

# if(CMAKE_CUDA_COMPILER)
# message(STATUS "Find CUDA !")
# else()
# message(FATAL_ERROR "Not found CUDA!")
# endif()

# 可以通过检查 CMAKE_CUDA_COMPILER（CMake 3.11 之前没有）来查看 CUDA 开发包是否存在
message(STATUS "There is CUDA toolkits: ${CMAKE_CUDA_COMPILER}")

# 可以检查 CMAKE_CUDA_COMPILER_ID（对于 nvcc，其值为 "NVIDIA",Clang将在CMake 3.18支持
message(STATUS "The CUDA toolkits is: ${CMAKE_CUDA_COMPILER_ID}")

# 可以用 CMAKE_CUDA_COMPILER_VERSION 检查 CUDA 版本
message(STATUS "The CUDA Version is: ${CMAKE_CUDA_COMPILER_VERSION}")

# 设置 nvcc 编译选项
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" --resource-usage)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;-G;-g)
# 自动检测并适应当前 GPU 架构
# set(CMAKE_CUDA_ARCHITECTURES OFF)
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 -Xptxas -dlcm=ca")
# set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")

# -------------------------------------------
# external 第三方库

# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
# add_subdirectory(00_OvertureCProgrammers)
add_executable(00_HelloCuda 00_HelloCuda.cu)
add_executable(01_ThreadModel 01_ThreadModel.cu)
add_executable(02_MatrixSum 02_MatrixSum.cu)
add_executable(03_CudaErrorCheck 03_CudaErrorCheck.cu)
add_executable(04_RuntimeEvent 04_RuntimeEvent.cu)
add_executable(05_QueryDeviceInfo 05_QueryDeviceInfo.cu)
add_executable(06_OrganizeThreads 06_OrganizeThreads.cu)
add_executable(07_ResisterMemory 07_ResisterMemory.cu)
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
ADD_DEFINITIONS("--resource-usage")
add_executable(08_GlobalMemory 08_GlobalMemory.cu)
add_executable(09_SharedMemory 09_SharedMemory.cu)
add_executable(10_ConstantMemory 10_ConstantMemory.cu)
add_executable(11_CacheMemory 11_CacheMemory.cu)
add_executable(12_ReduceCompute 12_ReduceCompute.cu)
