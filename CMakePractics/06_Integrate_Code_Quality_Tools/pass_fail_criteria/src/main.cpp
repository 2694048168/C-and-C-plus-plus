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

#include <iostream>

int main(int argc, const char **argv)
{
    if (argc == 1)
    {
        std::cout << "Error: No arguments passed\n";
    }
    else if (argc % 2 == 0)
    {
        std::cout << "Success: Odd number of arguments" << std::endl;
    }
    else
    {
        std::cout << "Warning: Even number of arguments" << std::endl;
    }

    return argc % 2;
}
