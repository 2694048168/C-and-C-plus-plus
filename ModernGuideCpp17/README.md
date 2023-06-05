> TODO list, we can quickly check in [VSCode](https://code.visualstudio.com/) with extension [Todo Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree).

### quick start

**the compilers for C++**
- [LLVM Clang download](https://releases.llvm.org/)
- [GCC download](https://gcc.gnu.org/releases.html)
- [MinGW download](https://winlibs.com/)
- [VS2022 cl download](https://visualstudio.microsoft.com/zh-hans/vs/)


```shell
gcc --version
# gcc.exe (MinGW-W64 x86_64-ucrt-posix-seh) 12.3.0

g++ --version
# g++.exe (MinGW-W64 x86_64-ucrt-posix-seh) 12.3.0

clang --version
# clang version 16.0.0

clang++
# clang version 16.0.0

cl
# 用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.35.32217.1 版(VS2022)
```

> **you must to know the build process for C++, include the compile and link. More detail information at [here](https://2694048168.github.io/blog/#/PaperMD/cpp_env_test)**

```shell
g++ main.cpp -std=c++17 # main
g++ main.cpp -std=c++20
g++ main.cpp -std=c++23

clang++ main.cpp -std=c++17 # main
clang++ main.cpp -std=c++20
clang++ main.cpp -std=c++2b # for ISO C++23

cl main.cpp /std:c++17 /EHsc # main
```
