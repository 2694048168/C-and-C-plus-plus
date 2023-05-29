/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-05-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fmt/core.h>

#include <iostream>

/**
 * @brief the build(compile and link) with CMake
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "hello modern c++\n";

    fmt::print("Hello, modern C++ world via fmt library!\n");

    return 0;
}
