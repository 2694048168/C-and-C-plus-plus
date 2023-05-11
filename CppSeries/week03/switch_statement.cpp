/**
 * @file switch_statement.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the switch statement in C++.
 * @attention default case and break in switch statement.
 *
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    unsigned char input_key = 0;
    std::cout << "Please enter a character to start:" << std::endl;
    std::cin >> input_key;
    while (input_key != 'q')
    {
        switch (input_key)
        {
        case 'a':
        case 'A':
            std::cout << "Move left. Enter 'q' to quit." << std::endl;
            break;
        case 'd':
        case 'D':
            std::cout << "Move right. Enter 'q' to quit." << std::endl;
            break;
        
        default:
            std::cout << "Undefined key. Enter 'q' to quit." << std::endl;
            break;
        }
        std::cin >> input_key;
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ switch_statement.cpp
 * $ clang++ switch_statement.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */