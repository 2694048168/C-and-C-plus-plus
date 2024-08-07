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
- 用 Canny 算子检测图像轮廓
- 用霍夫变换检测直线
- 点集的直线拟合
- 提取连续区域
- 计算区域的形状描述子
- 检测图像中的角点
- 快速检测特征
- 尺度不变特征的检测
- 多尺度 FAST 特征的检测
- 局部模板匹配
- 描述并匹配局部强度值模式
- 用二值描述子匹配关键点
- 计算图像对的基础矩阵
- 用 RANSAC 算法匹配图像
- 计算两幅图像之间的单应矩阵
- 检测图像中的平面目标
- 相机标定
- 相机姿态还原
- 用标定相机实现三维重建
- 计算立体图像的深度
- 读取视频序列
- 处理视频帧
- 写入视频帧
- 提取视频中的前景物体
- 跟踪视频中的特征点
- 估算光流
- 跟踪视频中的物体
- 用最邻近局部二值模式实现人脸识别
- 通过级联 Haar 特征实现物体和人脸定位
- 用支持向量机和方向梯度直方图实现物体和行人检测

### Problem OpenCV3.x VS OpenCV4.x
- 25_calibrating_camera
- 29_visual_tracker_video
- 32_ML_train_SVM

----------------------------

## Introduction

OpenCV ([Open Source Computer Vision Library](https://opencv.org/)) 是一个开源的计算机视觉算法库，包含数百种计算机视觉算法。


### Installation in Linux

```shell
# 参考文档：https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html
# 参考文档：https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
# 参考文档：https://docs.opencv.org/master/db/d05/tutorial_config_reference.html
# 参考文档：https://docs.opencv.org/master/d0/d3d/tutorial_general_install.html
# 下载失败：https://blog.csdn.net/fzp95/article/details/109276633
# 下载失败：https://blog.csdn.net/valley2013/article/details/106911688
# 安装编译工具
sudo apt update && sudo apt install -y build-essential cmake git wget unzip

# dlib 相比于 OpenCV 的优点之一就是不需要依赖第三方库
# 安装依赖
sudo apt install -y libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libeigen3-dev libtbb-dev libgtk2.0-dev libgtk-3-dev 

sudo apt install pkg-config libcanberra-gtk-module libcanberra-gtk3-module
sudo apt update && sudo apt install ffmpeg

sudo apt install libgtk2.O-dev libavcodec-dev libavformat-dev libjpeg.dev libpng-dev libtiff-dev libtiff4.dev libswscale-dev libjasper-dev libcur14-openssl-dev libtbb2 libdc1394-22-dev

# 修复可能安装出错的依赖
sudo apt install -f

# 下载源码
# https://opencv.org/releases/
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip
tar -zxvf opencv.tar.gz


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
# 首先为系统添加 OpenCV 库
sudo gedit /etc/ld.so.conf.d/opencv.conf
# 添加内容
/usr/local/lib
# 使 OpenCV 配置文件生效
sudo ldconfig

# 然后配置 bash 环境变量(配置系统级别和用户级别都可以)
# 查看该文件是否存在（OPENCV_GENERATE_PKGCONFIG=YES参数保证此文件存在）
cat /usr/local/opencv4/lib/pkgconfig/opencv4.pc
# 把上面的文件添加到PKG_CONFIG_PATH
sudo vim /etc/profile.d/pkgconfig.sh
# 文件内容如下
# export PKG_CONFIG_PATH=/usr/local/opencv4/lib/pkgconfig:$PKG_CONFIG_PATH
# 激活文件
source /etc/profile
# 验证配置，如果不报错则说明正常
pkg-config --cflags --libs opencv4
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

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.3/modules -DOPENCV_ENABLE_NONFREE=ON -DCMAKE_BUILD_TYPE=Release -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_CUDA=OFF -DWITH_QT=OFF -DWITH_GTK=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_FFMPEG=ON ../opencv-4.5.3

cmake --build . -j8

```


### Using OpenCV with g++ and CMake

1. 利用安装好的 OpenCV 库进行编程
2. 编写 CMakeLists.txt 构建文件
3. 利用 CMake 进行构建并编译
4. 查看编写程序的效果


```shell
# CMakeListx.txt CMake 构建文件样例
# cmake needs this line
cmake_minimum_required(VERSION 3.10)

# Define project name
project(opencv_example_project)

set(OpenCV_DIR /home/weili/opencv4-install/)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(opencv_example example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_example PRIVATE ${OpenCV_LIBS})


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
