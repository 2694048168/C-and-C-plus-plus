/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @brief the header file in C++ and multiple file compiler and link in C++.
 *   the compiler error | the linker error | runtime error in C++.
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>

// the header file in C++ just copy in Pre-processing phase by preprocessor.
// how to handle the search path for header file.
#include "arithmetic/mymul.hpp"

/**
 * @brief main function and the entry of program.
 */
int main(int argc, char const *argv[])
{
    /* Step 1. the basic input stream in C++.
    --------------------------------------------- */ 
    int arg_1;
    int arg_2;

    std::cout << "Pick two integers: ";
    std::cin >> arg_1;
    std::cin >> arg_2;

    // the paramters and the arguments in C++.
    int result = mul_number(arg_1, arg_2);
    std::cout << "the result of two integers: " << result << std::endl;

    /* Step 2. the runtime error in C++, such as the segmentation fault
     because of the invalid memory accsss
     (reading or writing some unpermitted memory regions).
    ------------------------------------------------------ */ 
    int *ptr = nullptr;
    int seed_random = 42;
    ptr = &seed_random;
    // ptr[0] = 42; /* invalid access */

    std::cout << "ptr pointer access: " << ptr[0] << std::endl;
    std::cout << "ptr pointer access: " << *ptr << std::endl;

    return 0;
}


/** Build(compile and link) commands via command-line.
 * 
 * $ clang++ main.cpp arithmetic/mymul.cpp
 * $ clang++ main.cpp arithmetic/mymul.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac 
 * 
*/