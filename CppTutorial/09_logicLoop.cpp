/**
 * @file 09_logicLoop.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中执行逻辑之循环执行
 * @version 0.1
 * @date 2024-03-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdlib>
#include <iostream>

// 左闭右闭区间
inline int getRand(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}

// ====================================
int main(int argc, const char **argv)
{
    // 设置随机数种子, 并生成一个随机数
    std::srand(time(0));
    std::cout << "======== 随机生成12个随机数 ========\n";
    for (size_t idx = 0; idx < 12; ++idx)
    {
        int random = getRand(1, 100);
        std::cout << "第 " << idx + 1 << " 一个随机数: " << random << '\n';
    }
    std::cout << "==================================\n";

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\09_logicLoop.cpp -std=c++23
// g++ .\09_logicLoop.cpp -std=c++23
