/**
 * @file float.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Floating-Point Numbers and stored on memory in C++.
 * @attention float-point random sample value resulting precision problem.
 *
 */

#include <iostream>
#include <ios>     /* std::fixed */
#include <iomanip> /* std::setprecision */
#include <cmath>   /* std::fabs() */
#include <cfloat>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. Floating-Point Numbers and stored on memory in C++.
    -------------------------------------------------------------- */
    float float_value1 = 1.2f;
    float float_value2 = float_value1 * 1000000000000000.f;

    std::cout << std::fixed << std::setprecision(15) << float_value1 << "\n";
    std::cout << std::fixed << std::setprecision(1) << float_value2 << std::endl;
    std::cout << std::endl;

    /* Since there are precision errors for floating-point numbers,
     using == to compare two floating-point numbers is a bad choice.

     If the difference between two numbers is less than a very small number,
     such as FLT_EPSILON or DBL_EPSILON for float and double in C++11
     on the header <cfloat>, respectively, we can think they are equal.
    ------------------------------------------------------------------- */
    float value_1 = 3.14f;
    float value_2 = 3.14f;
    std::cout << std::fixed << std::setprecision(16) << value_1 << "\n";
    std::cout << std::fixed << std::setprecision(16) << value_2 << std::endl;
    if (std::fabs(value_1 - value_2) < FLT_EPSILON)
    {
        std::cout << "the two folat value is equal." << std::endl;
    }
    else
    {
        std::cout << "the two folat value is not equal." << std::endl;
    }
    std::cout << std::endl;

    /* Step 2. as followint example demonstrates if a large number is
     added to a small one, the result will be the same as the large one.
     It is caused by the precision problem in C++.
    ----------------------------------------------- */
    float f1 = 23400000000.f;
    float f2 = f1 + 10.f;

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout << "f1 = " << f1 << std::endl;
    std::cout << "f2 = " << f2 << std::endl;
    std::cout << "f1 - f2 = " << f1 - f2 << std::endl;
    std::cout << "(f1 - f2 == 0) = " << (f1 - f2 == 0) << std::endl;

    // INF and NAN in division operations
    float var_1 = 2.0f / 0.0f; /* 除以无穷小, 结果为无穷大 */
    float var_2 = 0.0f / 0.0f;
    std::cout << "the divisor is zero: " << var_1 << std::endl;
    std::cout << "the divisor is zero: " << var_2 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ float.cpp
 * $ clang++ float.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */