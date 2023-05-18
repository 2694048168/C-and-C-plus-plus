/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-18
 * @version 0.1.1
 * @copyright Copyright (c) 2023
 *
 * @brief the tutorial of CMake for C++
 * @attention 
 */

#include <iostream>

// module-style
#include "calc/mycalc.hpp"

/**
 * @brief main function 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv)
{
    std::cout << "Hello CMake for CPP World.\n";

    int integer_1 = 42;
    int integer_2 = 24;

    std::cout << "the add result: " << myadd(integer_1, integer_2) << std::endl;
    std::cout << "the sub result: " << mysub(integer_1, integer_2) << std::endl;
    std::cout << "the div result: " << mydiv(integer_1, integer_2) << std::endl;
    std::cout << "the mul result: " << mymul(integer_1, integer_2) << std::endl;
    
    return 0;
}