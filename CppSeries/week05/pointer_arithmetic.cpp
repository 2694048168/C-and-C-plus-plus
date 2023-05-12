/**
 * @file pointer_arithmetic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief pointer arithmetic
 * @attention ++pointer and --pointer with element-wise address
 *
 */

#include <iostream>

// marco must be in one line.
#define PRINT_ARRAY(array, n) \
for (int idx = 0; idx < (n); idx++) \
    std::cout << "array[" << idx << "] = " << (array)[idx] << "\n";

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    int numbers[4] = {0, 1, 2, 3};
    PRINT_ARRAY(numbers, 4)

    int * p = numbers + 1; // point to the element with value 1
    p++; // point to the element with value 2

    std::cout << "numbers = " << numbers << std::endl;
    std::cout << "p = " << p << std::endl;

    *p = 20; //change number[2] from 2 to 20
    *(p-1) = 10; //change number[1] from 1 to 10
    p[1] = 30; //change number[3] from 3 to 30

    PRINT_ARRAY(numbers, 4)

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ pointer_arithmetic.cpp
 * $ clang++ pointer_arithmetic.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */