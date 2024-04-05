## 使用 CMake 构建复杂现代 C++ 项目工程

### Organization of Modern C++ Project
```
. CMakeProjectTemplate
|—— src
|   |—— main.cpp
|   |—— CMakeLists.txt
|   |—— MathLib
|   |—— |—— CMakeLists.txt
|   |—— |—— MathLib.hpp
|   |—— |—— MathLib.cpp
|   |—— PrintModule
|   |—— |—— CMakeLists.txt
|   |—— |—— ModuleLib.hpp
|   |—— |—— ModuleLib.cpp
|—— external
|   |—— |—— log
|—— bin
|—— test
|—— docs
|—— build
|—— cmake_build.sh
|—— .gitignore
|—— .clang-format
|—— LICENSE
|—— README.md
```

> First off Debug/Release are called configurations in cmake.

If you are using a single configuration generator (Ninja/Unix-Makefiles) you must specify the CMAKE_BUILD_TYPE. Like this:
```shell
# Configure the build
cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Debug
# Actually build the binaries
cmake --build build/

# Configure a release build
cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Release
# Build release binaries
cmake --build build/
```

For multi-configuration generators it's slightly different (Ninja Multi-Config, Visual Studio). Like this:
```shell
# Configure the build
cmake -S . -B build
# cmake -S . -B build -G Ninja

# Build debug binaries
cmake --build build --config Debug

# Build release binaries
cmake --build build --config Release
```

scp 是 secure copy 的缩写, scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令, 复制文件和目录.
```shell
# 1、从本地复制到远程
scp -r -p -v local_folder remote_username@remote_ip:remote_folder

# 2、从远程复制到本地
scp -r -p -v remote_username@remote_ip:remote_folder local_folder

# 如果远程服务器防火墙有为scp命令设置了指定的端口，需要使用 -P 参数来设置命令的端口号
# scp 命令使用端口号 4588
scp -P 4588 -r remote_username@remote_ip:remote_folder local_folder
```
