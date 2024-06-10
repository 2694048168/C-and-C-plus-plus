/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

void hello_world(const std::string &name)
{
    std::cout << name
              << ", Welcome Modern C++"
                 " and Modern CMake Practices\n";
}

// -----------------------------------
int main(int argc, const char **argv)
{
    hello_world("Ithaca");

    return 0;
}
