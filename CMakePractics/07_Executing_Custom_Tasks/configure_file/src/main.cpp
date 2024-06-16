/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "version.h"

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "This file was built on the git revision: " << CMAKE_BEST_PRACTICES_VERSION << std::endl;

    return 0;
}