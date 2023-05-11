/**
 * @file const_variable.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Constant Numbers and Variables | Data Type Conversions
 * @attention If a variable/object is const-qualified, it cannot be modified.
 *  It must be initialized when you define it.
 *
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. variable/object is const-qualified in C++.
    ----------------------------------------------------- */
    const float pi = 3.1415926;
    // pi += 1; /* invalid or error */
    std::cout << std::endl;

    /* Step 2. Data Type Conversions in C++.
     Attention: implicit and explicit conversion.
    ----------------------------------------------- */
    unsigned char char_value1 = 255;
    unsigned char char_value2 = 1;
    int num = char_value1 + char_value2;
    int num_result = (int)char_value1 + (int)char_value2;
    std::cout << "the result of char: " << num << std::endl;
    std::cout << "the result of char: " << num_result << std::endl;

    int implicit_conversion = 1.2f + 3.4;
    // ---> 1.2 + 3.4 ---> 4.6 ---> 4
    int explicit_conversion = (int)((double)1.2f + 3.4);

    float float_result = 17 / 5; /* ---> 3.0f, not 3.4f */
    float float_result2 = 17 / 5.f;
    std::cout << "the result: " << float_result << std::endl;
    std::cout << "the result: " << float_result2 << std::endl;
    std::cout << std::endl;

    /* Step 3. auto Type in C++11.
        Assignment Operators
    ----------------------------- */
    auto int_var = 42;
    auto float_var = 3.14f;
    auto double_var = 3.14;
    std::cout << "the dat type: " << typeid(int_var).name() << std::endl;
    std::cout << "the dat type: " << typeid(float_var).name() << std::endl;
    std::cout << "the dat type: " << typeid(double_var).name() << std::endl;
    std::cout << std::endl;

    std::cout << "the result of ++int_var expression: "
              << ++int_var << std::endl;

    std::cout << "the result of ++int_var expression: "
              << int_var++ << std::endl;

    std::cout << "the value of int_var variable: "
              << int_var << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ const_variable.cpp
 * $ clang++ const_variable.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */