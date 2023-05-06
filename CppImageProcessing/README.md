## Image Processing with Cpp

> Image Processing Opertors and Algorithms with c++ via OpenCV libarary.

**Overview**
- [Quick Start](#quick-start)
- [Useful Links](#useful-links)
- [Organization of Project](#organization-of-project)


### Quick Start

```shell
# step 1: clone the repo. and into the folder 'CppImageProcessing'
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CppImageProcessing

# install C++ package manager 'vcpkg'
# step 2: modify the 'vspkg' install path in the top 'CMakeLists.txt' file.
set(CMAKE_TOOLCHAIN_FILE "[path to vcpkg]/scripts/buildsystems/vcpkg.cmake")

# it may be time consumming in the first time build,
# because of downloading and building the OpenCV library.
# CMake configuration and build(compiler and linker)
cmake -B build
cmake --build build

# enter into 'bin' and running the demo example, such as 'RandomText'
cd bin
./RandomText
./CudaOpenCV
```

### Useful Links
- [OpenCV](https://github.com/opencv/opencv)
- [OpenCV contrib](https://github.com/opencv/opencv_contrib)
- [OpenCV imgproc module](https://docs.opencv.org/4.7.0/d7/da8/tutorial_table_of_content_imgproc.html)
- [vcpkg](https://vcpkg.io/en/getting-started.html)
- [CMake](https://cmake.org/download/)
- [Git](https://git-scm.com/downloads)

### Organization of Project
```
. CppImageProcessing
|—— hello_start
|   |—— main.cpp
|   |—— CMakeLists.txt
|—— build
|   |—— |—— vcpkg_installed
|   |—— |—— |—— x64-windows
|   |—— |—— |—— |—— bin
|   |—— |—— |—— |—— lib
|   |—— |—— |—— |—— include
|—— CMakeLists.txt
|—— vcpkg.json
|—— bin
|—— lib
|—— images
|—— README.md
```
