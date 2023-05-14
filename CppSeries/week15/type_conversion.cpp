/**
 * @file type_conversion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief type conversion and Type Cast Operators in C++
 * @attention C-style and C++ style
 *
 */

#include <iostream>
#include <cstdio>
#include <string>


/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    std::cout << "------------ const_cast --------------\n";
    int value1 = 100;
    const int value2 = 200;
    std::cout << "value1 = " << value1 << std::endl;
    std::cout << "value2 = " << value2 << std::endl;

    int * pv1 = const_cast<int *>(&value1);
    int * pv2 = const_cast<int *>(&value2);

    (*pv1)++;
    (*pv2)++;

    std::cout << "value1 = " << value1 << std::endl;
    std::cout << "value2 = " << value2 << std::endl;
    
    int& v2 = const_cast<int&>(value2);
    v2++;
    std::cout << "value2 = " << value2 << std::endl;

    std::cout << "*pv2 = " << (*pv2) << std::endl;
    std::cout << "v2 = " << v2 << std::endl;

    std::cout << "------------ reinterpret_cast --------------\n";
    int i = 18;
    float * p1 = reinterpret_cast<float *>(i); // static_cast will fail
    int * p2 = reinterpret_cast<int*>(p1);

    printf("p1=%p\n", p1);
    printf("p2=%p\n", p2);
    
    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ type_conversion.cpp
 * $ clang++ type_conversion.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */