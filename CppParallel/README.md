## C++ Parallel and Concurrency 并行和并发编程实战

> Talk is cheap, show me the code.

### features
- Parallel and Concurrency
    - [x] [OpenMP](https://www.openmp.org/) 需要编译器(compiler)支持, [TBB(Intel)](https://github.com/uxlfoundation/oneTBB)需要下载运行库
    - [x] Modern C++ thread and Modern CPU multi-core processor
    - [x] Modern C++ thread and Double Buffer technology
    - [x] Thread Concurrency and C++ lock technology
    - [x] [SIMD intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)(AVX512/AVX2/SSE2/SSE) and Vectorized Compute
    - [x] [thread affinity](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask) and CPU-core
    - [x] [memory reordering](https://aaron-ai.com/docs/memory_reordering_simple_analysis/) : 只是在实际实践的过程中往往会使用 memory reordering 的特性来保证多线程操作时的线程安全而已; 所谓 memory reordering，本质上就是编译器和 CPU 对单线程中指令执行顺序进行的优化.

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
