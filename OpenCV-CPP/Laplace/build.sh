#!/bin/sh
#
# 支持 Windows 下 MinGW and Git bash
# 支持 Linux 下 GCC and bash
#

# checking OS
if [[ "$OSTYPE" == "msys" ]];
then
    # Windows NT操作系统
    echo "Info: --the OS is Windows."
    rm -rf *
    if [[ $? != 0 ]]
    then
        echo "Error  --清除编译缓存命令有错误，请查看日志提示！"
	    exit 0
    else
        echo "Info  --Clean the build files!"
    fi

    echo "Info  --the CMake Version : $(cmake --version)"
    cmake .. -G "MinGW Makefiles"
    if [[ $? != 0 ]]
    then
        echo "Error  --CMake 失败，请查看日志提示！"
	    exit 0
    else
        echo "Info  --CMake OK!"
    fi

    echo "Info  --the make of MinGW Version : $(mingw32-make --version)"
    mingw32-make
    if [[ $? != 0 ]]
    then
        echo "Error  --Make 失败，请查看日志提示！"
	    exit 0
    else
        echo "Info  --Make OK!"
    fi
    # ending with Windows OS

elif [[ "$OSTYPE" == "linux-gnu" ]];
then
    # GNU/Linux操作系统
    echo "Info: --the OS is $(uname)."
    rm -rf *
    if [[ $? != 0 ]]
    then
        echo "Error  --清除编译缓存命令有错误，请查看日志提示！"
	    exit 0
    else
        echo "Info  --Clean the build files!"
    fi

    echo "Info  --the CMake Version : $(cmake --version)"
    cmake .. -G "Unix Makefiles"
    if [[ $? != 0 ]]
    then
        echo "Error  --CMake 失败，请查看日志提示！"
	    exit 0
    else
        echo "Info  --CMake OK!"
    fi

    echo "Info  --the make of GCC Version : $(make --version)"
    make
    if [[ $? != 0 ]]
    then
        echo "Error  --Make 失败，请查看日志提示！"
	    exit 0
    else
        echo "Info  --Make OK!"
    fi
    # ending with Unix/Linux OS
fi # ending with checking OS
