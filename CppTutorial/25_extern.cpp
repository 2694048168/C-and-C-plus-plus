/**
 * @file 25_extern.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-09-18
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "25_add.h"

#include <functional>
#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    auto res = add(21, 21);

    auto out = std::ref(std::cout << "Result from C code: " << res);
    out.get() << ".\n";

    return 0;
}

// 1. 应先使用 gcc/clang 编译 C 语言的代码
// gcc -c 25_add.c
// clang -c 25_add.c
// 2. 编译出 25_add.o 文件，再使用 clang++ 
// 将 C++ 代码和 .o 文件链接起来（或者都编译为 .o 再统一链接）：
// clang++ 25_extern.cpp 25_add.o -std=c++20 -o demo.exe
