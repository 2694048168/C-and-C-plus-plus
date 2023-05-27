/**
 * @file clearIO_buffer.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief clearing the input buffer in C++. Explain why not clearing
 * the input buffer causes undesired outputs. '\n'?
 *
 * https://www.geeksforgeeks.org/clearing-the-input-buffer-in-cc/
 * https://www.geeksforgeeks.org/memory-layout-of-c-program/
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    int a;
    char ch[80];
     
    // Enter input from user - 4 for example
    std::cin >> a;
     
    // Get input from user - "GeeksforGeeks" for example
    std::cin.getline(ch,80);
     
    // Prints 4
    std::cout << a << std::endl;
     
    // Printing string : This does not print string
    std::cout << ch << std::endl;

    return 0;
}