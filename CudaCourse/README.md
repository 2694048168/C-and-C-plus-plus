## 现代 Cuda 编程：从入门到实践

> CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.

[Cuda Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

### Features
- Cuda 开发基本环境配置
```shell   
# WSL2 on Windows
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
- [Cuda Download](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-archive)

### Quick Start
```shell
# step 1. download and install Visual Studio Code(VSCode)
# step 2. download and install VS/GCC/Clang Toolchain
# step 3. download and install CMake and Ninja
# step 4. install extension in VSCode
# step 5. install CUDA
# $env:CUDATOOLKITDIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<yourversion>"
# 针对学习最佳方式是 Linux 操作系统, WSL也可以(直接支持使用Windows主机显卡资源)
# CUDA 环境配置更加便捷, command line 方式

mkdir code && cd code
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd CudaLearning

# build and compile on Linux
cmake -S . -B build -G Ninja
cmake --build build --config Release

# Windows OS
# 需要进入到 VS2022 的 Developer PowerShell 环境, 需要 cl 编译器的支持
nvcc 00_HelloCuda.cu

# .bashrc or .zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# nvprof or ncu 进行 CUDA 程序性能分析
nvprof bin/04_RuntimeEvent
sudo nvprof --unified-memory-profiling per-process-device bin/04_RuntimeEvent
nvprof --unified-memory-profiling per-process-device bin/04_RuntimeEvent
nvprof --unified-memory-profiling off bin/04_RuntimeEvent
ncu bin/04_RuntimeEvent
# https://developer.nvidia.com/blog/even-easier-introduction-cuda/
```

### Organization
```
. CudaLearning
|—— 00_OvertureCProgrammers
|   |—— CMakeLists.txt
|—— CMakeLists.txt
|—— bin
|—— build
|—— .gitignore
|—— .clang-format
|—— README.md
```

```shell
# 1、从本地复制到远程
scp -r -p -v local_folder remote_username@remote_ip:remote_folder

# 2、从远程复制到本地
scp -r -p -v remote_username@remote_ip:remote_folder local_folder

# 如果远程服务器防火墙有为scp命令设置了指定的端口，需要使用 -P 参数来设置命令的端口号
# scp 命令使用端口号 4588
scp -P 4588 -r remote_username@remote_ip:remote_folder local_folder
```
