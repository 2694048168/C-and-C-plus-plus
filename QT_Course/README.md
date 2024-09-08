## Modern C++ and QT

> Modern C++ and QT creating software applications, from planning and design to development, testing, and future-proofing your products.

```shell
# install the QT5 and QT6
# VSCode + CMake + Qt Official extension
# MSVC无法自动生成 compile_commands.json

# VSCode + CMake + Qt Support extension + cland
# 采用 Ninja 构建编译 自动生成 compile_commands.json
cmake -S . -B build -G "Ninja"
cmake --build build

# C:\Qt\5.15.2\mingw81_64\bin
windeployqt 00_dev_env.exe
```

### Qt基础
- Qt认识入门 **00_dev_env.cpp**  

### Qt控件

### Qt事件

### 套接字通信

### Qt线程

### 数据库

### 打包部署

