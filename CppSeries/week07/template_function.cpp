/**
 * @file template_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the function template in C++
 * @attention Implicitly or Explicitly instantiate
 *
 */

#include <iostream>
#include <typeinfo>

template <typename T>
T mysum(T x, T y)
{
    std::cout << "The input type is "
              << typeid(T).name() << "\n";
    return x + y;
}

// Explicitly instantiate
template double mysum<double>(double, double);

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    // Implicitly instantiate
    std::cout << "sum = " << mysum(1, 2) << std::endl;
    std::cout << "sum = " << mysum<float>(1.1f, 2.2f) << std::endl;

    std::cout << "sum = " << mysum(1.1, 2.2) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ template_function.cpp
 * $ clang++ template_function.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */