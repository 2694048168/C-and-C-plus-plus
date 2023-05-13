### quick start

**Build(compile and link) commands via command-line.**
```shell
# if you install the OpenCV library, and then as following:
clang++ mat_example.cpp
clang++ mat_example.cpp -std=c++17
./a.exe # on Windows
./a.out # on Linux or Mac

# if you NOT install the OpenCV library, and then as following:
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
vcpkg install opencv4:x64-windows
cmake -B build -G Ninja -A x64 -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake

# Attention: 注意 host 的CPU架构 (ARM X64), 编译指定的 OpenCV 库
```
