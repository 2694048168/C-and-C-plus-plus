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

// ------------------------
int main(int, char **)
{
    int *i = new int[10];
    delete[] i;
    std::cout << i[0] << "\n";

    return 0;
}