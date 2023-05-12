/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the parameters pass by pointer in function
 * @attention
 *
 */

#include <iostream>

#include "math/mymath.hpp"


int main(int argc, char const *argv[])
{
    float arr1[8]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    float *arr2 = nullptr;

    float sum1 = arraySum(arr1, 8);
    float sum2 = arraySum(arr2, 8);

    std::cout << "The result1 is " << sum1 << std::endl;
    std::cout << "The result2 is " << sum2 << std::endl;

    return 0;
}


/** Build(compile and link) commands via command-line or CMake.
 *
 * $ clang++ main.cpp math/mymath.cpp
 * $ clang++ main.cpp math/mymath.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */