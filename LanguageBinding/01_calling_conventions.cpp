/**
 * @file 01_calling_conventions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// #define CALLING_CONVENTIONS __stdcall
#define CALLING_CONVENTIONS __cdecl

int CALLING_CONVENTIONS addNum(int x, int y)
{
    return x + y;
}

int CALLING_CONVENTIONS subNum(int x, int y)
{
    return x - y;
}

// ====================================
int main(int argc, const char **argv)
{
    auto ret = addNum(12, 24);
    std::cout << "The add of 12 and 24 ---> " << ret << '\n';

    auto ret_ = subNum(24, 12);
    std::cout << "The sub of 24 and 12 ---> " << ret_ << '\n';

    return 0;
}
