## the multi-threading in Modern C++

> We must be aware that different compilers may have differences in their implementation of the C++ standard, including the GCC(g++), Clang(clang++), MSVC(cl), Apple Clang, Intel C++, IBM XLC++, Nvidia nvcc, Nvidia HPC C++, Embarcadero C++ Builder.

Features:

- [x] Compiler supports g++, clang++ and MSVC(cl)
- [x] clang-format support
- [] CMake support
- [x] Creating threads
- [x] Using function pointers, functors, and lambda functions
- [x] Futures, promises, and async tasks
- [x] Mutex and Locks
- [x] Conditional Variables
- [] memory order and progaram order
- [] atomic operator
- [] [google benchmark](https://github.com/google/benchmark)

### the order of files to learning

1. main.cpp
2. vector_threads.cpp
3. functors_thread.cpp
4. functors_thread_unique_ptr.cpp
5. lambda_thread.cpp
6. async_main.cpp
7. async_main_lambda.cpp
8. mutex
    - race_condition_main.cpp
    - no_lock_main.cpp
    - shared_mutex_main.cpp
    - mutex_seq_cst_main.cpp
    - lock_guard_main.cpp
    - lock_guard_multiple_mutex_main.cpp
    - lock_unlock_main.cpp
    - unique_lock_main.cpp
9. conditional_variable
    - producer_consumer_conditional_var_main.cpp
    - producer_consumer_conditional_var_simple.cpp
    - producer_consumer_lock_main.cpp
10. memory_order
    - *.cpp
11. atomic_operator
    - *.cpp

> Are Atomic Operations Fast?

### compiler command-line
```shell
# the Clang compiler
clang++ main.cpp -std=c++17

# the GCC compiler, include MinGW
g++ main.cpp -std=c++17

# the MSVC(cl) compiler
cl main.cpp /std:c++17 /EHsc
```

> [reference C++ Multi Threading tutorial](https://www.bilibili.com/video/BV1zF411b7Bv/)