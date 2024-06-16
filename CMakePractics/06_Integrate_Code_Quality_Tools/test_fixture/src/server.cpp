/**
 * @file server.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// -------------------------------
int main(int argc, char **argv)
{
    if (argc > 1)
    {
        std::cout << "Teardown\n";
    }
    else
    {
        std::cout << "setup\n";
    }

    return 0;
}
