# OpencCV_CPP_Linux

## OpenCV 2.x API，C++

### OpenCV for CPlusPlus on Ubuntu Linux dist.
### [Reference Docs](https://docs.opencv.org/)

## What we can learning?

- 熟练使用 C++ 进行程序设计，计算机视觉实战
- 熟练学会 Linux Ubuntu 发版本的常用终端命令以及 shell 脚本
- 熟练使用计算机视觉 OpenCV 库
- 熟练使用 CMake 进行编译和链接 C++ 工程项目

## Overview

- 如何安装 OpenCV 库
- 如何装载、显示和存储图像
- 深入理解 cv::Mat 数据结构
- 定义 ROI（感兴趣区域）
- 访问像素值
- 用指针扫描图像
- 用迭代器扫描图像
- 编写高效的图像扫描循环
- 扫描图像并访问相邻像素
- 实现简单的图像运算
- 图像重映射
- 用策略设计模式比较颜色
- 用 GrabCut 算法分割图像
- 转换颜色表示法
- 用色调、饱和度和亮度表示颜色
- 计算图像直方图
- 利用查找表修改图像外观
- 直方图均衡化
- 反向投影直方图检测特定图像内容
- 用均值平移算法查找目标
- 比较直方图搜索相似图像
- 用积分图像统计像素
- 用形态学滤波器腐蚀和膨胀图像
- 用形态学滤波器开启和闭合图像
- 在灰度图像中应用形态学运算
- 用分水岭算法实现图像分割
- 用 MSER 算法提取特征区域
- 用低通滤波器进行图像滤波
- 用滤波器进行缩减像素采样
- 用中值滤波器进行图像滤波
- 用定向滤波器检测边缘
- 计算图像的拉普拉斯算子

----------------------------

## Introduction

OpenCV ([Open Source Computer Vision Library](https://opencv.org/)) 是一个开源的计算机视觉算法库，包含数百种计算机视觉算法。


### Installation in Linux

```shell
# 安装编译工具
sudo apt update && sudo apt install -y build-essential cmake git wget unzip pkg-config
# 安装依赖
sudo apt install -y libgtk2.0-dev
sudo apt install -y libgtk-3-dev
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev
sudo apt install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libjasper-dev libdc1394-22-dev
sudo apt install -y libpng-dev libopenexr-dev libtiff-dev libwebp-dev libtbb2 libtbb-dev libjpeg-dev

# 修复可能安装出错的依赖
sudo apt install -f

# 下载源码
# https://opencv.org/releases/
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip

# 配置并构建源码
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master

# 配置编译选项
# 1. OpenCV contrib 路径： -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules
# 2. 开启使用第三方专利模块： -DOPENCV_ENABLE_NONFREE=ON
# 3. 安装路径设置(本质就是拷贝编译好的库到指定路径下)： -DCMAKE_INSTALL_PREFIX=../third_party/OpenCV4/
# 4. 指定编译的版本： -DCMAKE_BUILD_TYPE=Release

# 5. 使用 pkg-config 进行配置，使得可以直接使用 g++编译(利用cmake不需要): -DOPENCV_GENERATE_PKGCONFIG=NO
# -----------------------------------------------
# 查看该文件是否存在（OPENCV_GENERATE_PKGCONFIG=YES参数保证此文件存在）
cat /usr/local/opencv4/lib/pkgconfig/opencv4.pc
# 把上面的文件添加到PKG_CONFIG_PATH
sudo vim /etc/profile.d/pkgconfig.sh
# 文件内容如下
# export PKG_CONFIG_PATH=/usr/local/opencv4/lib/pkgconfig:$PKG_CONFIG_PATH
# 激活文件
source /etc/profile
# 验证配置，如果不报错则说明正常
pkg-config --libs opencv4
# -----------------------------------------------

# 6. 是否需要 Python 版本的库： -DINSTALL_PYTHON_EXAMPLES=OFF
# 7. 是否开启 CUDA 支持，该选项需要确保自己已安装显卡驱动和cuda： -DWITH_CUDA=OFF
# 8. 是否使用 QT: -DWITH_QT=OFF
# 9. 将 OpenCV所有库编译到一个文件中(针对VS)： -DBUILD_opencv_world=ON
# -------------------------------
# 根据自己的代理工具设置的本地代理端口进行设置， 当然也可以使用http_proxy和https_proxy两个变量设置http代理
export all_proxy=socks5://127.0.0.1:55555
export all_proxy=http_proxy://127.0.0.1:55555
curl ifconfig.me    # 通过返回的IP地址判断代理是否设置成功
cmake-gui       # 从设置了代理的终端启动 cmake-gui 或者直接在此终端开始命令行编译
# -------------------------------
# CMake 编译 opencv各选项的含义
# https://blog.csdn.net/j_d_c/article/details/53365381
# -------------------------------

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules -DOPENCV_ENABLE_NONFREE=ON -DCMAKE_INSTALL_PREFIX=../third_party/OpenCV4/ -DCMAKE_BUILD_TYPE=Release -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_CUDA=OFF -DWITH_QT=OFF -DWITH_GTK=ON ../opencv-master

# --------------------------------
# 总核数 = 物理CPU个数 X 每颗物理CPU的核数
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数
# 查看物理CPU个数
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo| grep "cpu cores"| uniq
# 查看逻辑CPU的个数
cat /proc/cpuinfo| grep "processor"| wc -l
# 查看线程数
grep 'processor' /proc/cpuinfo | sort -u | wc -l 
# --------------------------------
cmake --build . -j8

# 安装 OpenCV 默认 /usr/local
sudo make install

# 卸载编译安装的 OpenCV
# cd build
# 执行卸载命令，此命令会删除安装时添加的所有文件，但是不处理文件夹
sudo make uninstall

```

```shell
# 安装 OpenCV shell 脚本 build_opencv.sh
mkdir -p build && cd build

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules -DOPENCV_ENABLE_NONFREE=ON -DCMAKE_BUILD_TYPE=Release -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_CUDA=OFF -DWITH_QT=OFF -DWITH_GTK=ON ../opencv-master

cmake --build . -j8

```


### Using OpenCV with g++ and CMake

1. 利用安装好的 OpenCV 库进行编程
2. 编写 CMakeLists.txt 构建文件
3. 利用 CMake 进行构建并编译
4. 查看编写程序的效果


```shell
# CMakeListx.txt CMake构建文件样例
cmake_minimum_required(VERSION 3.10)

project(project_name)

# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置所有待编译的源文件
file(GLOB SRC_FILE *.cpp)
message(STATUS ${SRC_FILE})
add_executable(project_name ${SRC_FILE})

# 链接
target_link_libraries(project_name ${OpenCV_LIBS})

# ----------------------------------
# 使用 CMake 进行编译和链接
rm -rf build
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build .

# ----------------------------------
# 运行程序
./build/project_name
```

### Git

```shell
git init
git config --global user.email "weili_yzzcq@163.com"
git config --global user.name "Wei Li"
# GitHub上 传文件不能超过1 00M 的解决办法, 修改为 500M
git config http.postBuffer 524288000
git config -l

# 利用 VSCode 智能插件保证路径不会写错 Path Intellisense
```


----------------------------

## About Author

### 掌中星辰转日月，手心苍穹天外天。
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;——云主宰苍穹

### Stay Hungry, Stay Foolish.
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;——Steve Jobs

- Mail：2694048168@qq.com
- Weibo：云主宰苍穹
- GitHub: https://github.com/2694048168/
- Gitee：https://gitee.com/weili_yzzcq/
