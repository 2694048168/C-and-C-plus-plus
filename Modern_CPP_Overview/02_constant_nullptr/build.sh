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