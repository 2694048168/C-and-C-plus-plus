/**
 * @file 04_LinkingCSymbols.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief Linking with C Symbols
 * 可以使用语言链接让 C 语言代码能识别 C++ 程序中的函数和变量,
 * *语言链接指示编译器生成对另一种目标语言友好的特定格式的符号.
 * 
 */

int main(int argc, const char **argv)
{
    printf("添加 extern C 语言链接\n");

    printf("Boost Python 在 C++ 和 Python 之间进行互操作\n");
    printf("https://github.com/pybind/pybind11\n");

    return 0;
}
