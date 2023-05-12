/**
 * @file function_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the function pointer in C++
 * @attention trick for callback function
 *
 */

#include <iostream>
#include <cmath>

float norm_l1(const float x, const float y); /* declaration */
float norm_l2(const float x, const float y); /* declaration */

// just a global pointer variable, and the ptr ---> function address
/* norm_ptr is a function pointer */
float (*norm_ptr)(const float x, const float y);

// norm_ref is a function reference
float (&norm_ref1)(const float x, const float y) = norm_l1; 
float (&norm_ref2)(const float x, const float y) = norm_l2; 

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. the function pointer variable
    ----------------------------------------- */ 
    norm_ptr = norm_l1; // Pointer norm_ptr is pointing to norm_l1
    std::cout << "L1 norm of (-3, 4) = "
              << norm_ptr(-3.0f, 4.0f) << std::endl;

    norm_ptr = &norm_l2; // Pointer norm_ptr is pointing to norm_l2
    std::cout << "L2 norm of (-3, 4) = "
              << (*norm_ptr)(-3.0f, 4.0f) << std::endl;

    std::cout << "---------------------------------\n";

    /* Step 1. the function pointer variable
    ----------------------------------------- */ 
    norm_ptr = norm_l1; // Pointer norm_ptr is pointing to norm_l1
    std::cout << "L1 norm of (-3, 4) = "
              << norm_ref1(-3.0f, 4.0f) << std::endl;

    norm_ptr = &norm_l2; // Pointer norm_ptr is pointing to norm_l2
    std::cout << "L2 norm of (-3, 4) = "
              << norm_ref2(-3.0f, 4.0f) << std::endl;
    std::cout << "---------------------------------\n";

    return 0;
}

float norm_l1(float x, float y)
{
    return std::fabs(x) + std::fabs(y);
}

float norm_l2(float x, float y)
{
    return std::sqrt(x * x + y * y);
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ function_pointer.cpp
 * $ clang++ function_pointer.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */