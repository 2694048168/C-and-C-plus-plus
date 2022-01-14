# Compile Command for 02_constant_nullptr

## Command line with g++ or clang++

```shell
# depending on how the compiler defines NULL 
g++ constant_nullptr.cpp -std=c++2a -o constant_nullptr

clang++ constant_nullptr.cpp -std=c++2a -o constant_nullptr
```

## Compile for CMake with CMakeLists.txt

```shell
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# 指定编译器
# set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe")
set(CMAKE_CXX_COMPILER "D:/mingw64/bin/clang++.exe")

# 设置编译选项
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++2a")
# 设置编译模式, Debug、Release
set (CMake_BUILD_TYPE "Debug")

project(constant_nullptr)

add_executable(${PROJECT_NAME} constant_nullptr.cpp)
```

## Compile for CMake build.sh

```shell
# MinGW compiler for windows
#!/bin/sh

rm -rf build
if [[ $? != 0 ]]
then
    echo "Error  --清除编译缓存命令有错误，请查看日志提示!"
	exit 0
else
    echo "INFO  --Clean the build files!"
fi

mkdir build && cd build
if [[ $? != 0 ]]
then
    echo "Error  --mkdir build 失败，请查看日志提示!"
	exit 0
else
    echo "INFO  --mkdir build OK!"
fi

cmake .. -G "MinGW Makefiles"
if [[ $? != 0 ]]
then
    echo "Error  --CMake 失败，请查看日志提示!"
	exit 0
else
    echo "INFO  --CMake OK!"
fi

mingw32-make
if [[ $? != 0 ]]
then
    echo "Error  --Make 失败，请查看日志提示!"
	exit 0
else
    echo "INFO  --Make OK!"
fi
```