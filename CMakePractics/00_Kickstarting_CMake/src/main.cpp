/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

void hello_world(std::string &name)
{
    std::cout << name
              << ", Welcome Modern C++"
                 " and Modern CMake Practices\n";
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::string name;
    std::cout << "Please enter your name: ";
    std::cin >> name;

    hello_world(name);

    return 0;
}
