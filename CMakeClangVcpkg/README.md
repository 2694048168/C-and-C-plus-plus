## the toolchain for Modern C++ in VSCode

> the toolchains for Modern C++, include CMake, vcpkg, Ninja, Clang and Git.

- [CMake download](https://cmake.org/download/)
- [vcpkg download](https://vcpkg.io/en/getting-started.html)
- [vcpkg.json example](https://learn.microsoft.com/zh-cn/vcpkg/reference/vcpkg-json)
- [vcpkg Browse packages](https://vcpkg.io/en/packages.html)
- [Ninja download](https://ninja-build.org/)
- [Clang download](https://releases.llvm.org/download.html)
- [Git download](https://git-scm.com/downloads)
- [VSCode](https://code.visualstudio.com/Download)
- [cland extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
- [CodeLLDB extension](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)
- [CMake extension](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
- [CMake Tools extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
- [CMake preset](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)

> the pure CMake configuration and build

```shell
cmake -B build -G Ninja

cmake --build build --config Release
# or build the 'Debug' version
cmake --build build --config Debug

./build/bin/main

cmake --build build --target clean
```

> the Preset CMake configuration and build

```shell
cmake --preset default

cmake --build --preset default --config Release
# or build the 'Debug' version
cmake --build --preset default --config Debug

./build/default/bin/main

cmake --build build/default --target clean
```

> the Build and Debugging in VSCode

```shell
# 可以借助 VSCode 提供的 CMake Tools 插件自动实现,
# 可能针对 LLVM-LLDM 提供的 debugging 不友好(^_^),
# 所以需要学会如何手动设置 debugging in VSCode(^_^)

# https://code.visualstudio.com/docs/editor/debugging
mkdir .vscode
touch .vscode/launch.json # for debugging with LLDB or GDB
touch .vscode/tasks.json # for build the binary file before debugging
# F9 打断点, F5 通过 launch.json 文件启动 debugging

# or 'Ctrl + Shift + P' and entry 'Debug: add configuration'
# or 'Ctrl + Shift + P' and entry 'Tasks: Configure Task'
```