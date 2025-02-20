## C++ Parallel and Concurrency 并行和并发编程实战

> Talk is cheap, show me the code.

### features
- Parallel and Concurrency
    - [x] OpenMP 需要编译器(compiler)支持, TBB(Intel)需要下载运行库
    - [x] Modern C++ thread and Modern CPU multi-core processor

### quick start
```shell
# clone the source code into a folder
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CppParallel

cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Debug
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Debug
cmake --build build --config Debug

cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Release
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build build --config Release

# executable the binary file
./hello
```

### reference
