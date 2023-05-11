/**
 * @file variables.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Data Types and Arithmetic Operators in C++.
 * @attention Fixed Width Integer Types since C++11 in <cstdint> header.
 *
 */

#include <iostream>
#include <typeinfo>
#include <cstdint>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. variables in C++ should be
     Declaration and Initialization
    ------------------------------------- */
    int var_1;
    int var_2;

    std::cout << "the variable value: " << var_1 << std::endl;
    std::cout << "the variable value: " << var_2 << std::endl;

    int var_1_init = 42;
    int var_2_init = 0;

    std::cout << "the variable value: " << var_1_init << std::endl;
    std::cout << "the variable value: " << var_2_init << "\n"
              << std::endl;

    /* Step 2. Data type in C++
     int folat double bool string
    ------------------------------------- */
    bool bool_data = true;
    int int_data = bool_data; /* implicit type conversion */
    std::cout << "the bool data: " << bool_data << std::endl;
    std::cout << "the bool data: " << std::boolalpha << bool_data << std::endl;
    std::cout << "the bool data: " << std::boolalpha << false << std::endl;
    std::cout << "the bool data: " << int_data << std::endl;
    std::cout << "the bool data: " << typeid(int_data).name() << "\n"
              << std::endl;

    char ch1 = 'C';
    char ch2 = 80;
    char ch3 = 0x50;
    char16_t ch4 = u'黎';
    char32_t ch5 = U'为';
    std::cout << ch1 << ":" << ch2 << ":" << ch3 << std::endl;
    std::cout << +ch1 << ":" << +ch2 << ":" << +ch3 << std::endl;
    std::cout << ch4 << ch5 << "\n"
              << std::endl;

    std::cout << "the bytes of bool: " << sizeof(bool) << std::endl;
    std::cout << "the bytes of short: " << sizeof(short) << std::endl;
    std::cout << "the bytes of int: " << sizeof(int) << std::endl;
    std::cout << "the bytes of long: " << sizeof(long) << std::endl;
    std::cout << "the bytes of long long: " << sizeof(long long) << std::endl;

    /* Another frequently used integer type is size_t.
     It is the type of the 'sizeof' operator. It can store the maximum size of a theoretically possible object of any type.
     We often need an integer variable to store the data size of
     a specific piece of memory.

     void* malloc(size_t size);
     void* new(int);
    --------------------------------- */
    std::cout << "the bytes of size_t: " << sizeof(size_t) << "\n"
              << std::endl;

    /* Step 3. Fixed Width(1 byte = 8 bit) Integer Types in C++

     int8_t | uint8_t | int16_t | uint16_t | int32_t | uint32_t
     int64_t | uint64_t | etc.

     There are some useful macros such as:
     INT_MAX | INT_MIN | INT8_MAX | UINT8_MAX | etc.
    ------------------------------------------------- */
    std::cout << "the betes of int8_t: " << sizeof(int8_t) << std::endl;
    std::cout << "the betes of uint8_t: " << sizeof(uint8_t) << std::endl;
    std::cout << "the betes of int16_t: " << sizeof(int16_t) << std::endl;
    std::cout << "the betes of uint16_t: " << sizeof(uint16_t) << std::endl;
    std::cout << "the betes of int32_t: " << sizeof(int32_t) << std::endl;
    std::cout << "the betes of uint32_t: " << sizeof(uint32_t) << std::endl;
    std::cout << "the betes of int64_t: " << sizeof(int64_t) << std::endl;
    std::cout << "the betes of uint64_t: " << sizeof(uint64_t) << std::endl;

    std::cout << "the value of INT_MIN macros: " << INT8_MIN << std::endl;
    std::cout << "the value of INT_MIN macros: " << INT8_MAX << std::endl;
    std::cout << "the value of INT_MIN macros: " << INT16_MIN << std::endl;
    std::cout << "the value of INT_MIN macros: " << INT16_MAX << std::endl;
    std::cout << "the value of INT_MIN macros: " << INT_MIN << std::endl;
    std::cout << "the value of INT_MIN macros: " << INT_MAX << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ variables.cpp
 * $ clang++ variables.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */