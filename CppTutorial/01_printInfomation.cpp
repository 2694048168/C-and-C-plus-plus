/**
 * @file 01_printInfomation.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 打印输出九九乘法口诀, 最基本的IO操作交互
 * @version 0.1
 * @date 2024-03-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>


/**
 * @brief 输出九九乘法口诀
 * 基本的思想, 通过双层循环来处理行列索引变化
 * 
 */
void printInfo()
{
    for (size_t i = 1; i < 10; ++i)
    {
        for (size_t j = 1; j <= i; ++j)
        {
            std::cout << i << " x " << j << " = " << i * j << "  ";
        }
        std::cout << std::endl;
    }
}

// ===================================
int main(int argc, const char **argv)
{
    std::cout << "=========== 输出九九乘法表 ===========\n";
    printInfo();
    std::cout << "==================================================\n";

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\01_printInfomation.cpp -std=c++23
// g++ .\01_printInfomation.cpp -std=c++23
