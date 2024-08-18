/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// #include "version.h"
#include "version.hpp"

#include <cstdio>

// -----------------------------
int main(int argc, char **argv)
{
    printf("This is output from code %s\n", PROJECT_VERSION);
    printf("Major version number: %i\n", PROJECT_VERSION_MAJOR);
    printf("Minor version number: %i\n", PROJECT_VERSION_MINOR);
    printf("Minor version number: %i\n", PROJECT_VERSION_PATCH);

    printf("Hello CMake world!\n");

    printf("The Project-Version: %s\n", PROGRAM_VERSION.c_str());

    return 0;
}
