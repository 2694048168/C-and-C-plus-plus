/**
 * @file overflow.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the Data Types and Expresentation Range in C++.
 * @attention the overflow in Arithmetic Operators and storged in C++.
 * 
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
*/
int main(int argc, char **argv)
{
    /* Step 1. variables in C++ should be
     Declaration and Initialization
    ------------------------------------- */
    int var_value1 = 56789;
    int var_value2 = 56789;
    // int var_result = var_value1 * var_value2;
    unsigned int var_result = var_value1 * var_value2;
    std::cout << "the result of arithmetic opertor: " << var_result << std::endl;

    return 0;
}


/** Build(compile and link) commands via command-line.
 * 
 * $ clang++ overflow.cpp
 * $ clang++ overflow.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac 
 * 
*/