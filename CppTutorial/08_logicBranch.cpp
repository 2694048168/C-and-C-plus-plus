/**
 * @file 08_logicBranch.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中执行逻辑之条件分支
 * @version 0.1
 * @date 2024-03-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <iostream>

// 左闭右闭区间
inline int getRand(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}

void guessRandomNumber(int randomNumber)
{
    int guessNumber = 0;
    std::cout << "Please enter an integer: ";
    std::cin >> guessNumber;

    if (guessNumber > randomNumber)
    {
        std::cout << "猜测的数值偏大了, 请重新猜测\n";
        guessRandomNumber(randomNumber);
    }
    else if (guessNumber < randomNumber)
    {
        std::cout << "猜测的数值偏小了, 请重新猜测\n";
        guessRandomNumber(randomNumber);
    }
    else
    {
        std::cout << "恭喜你, 神秘的数字被你猜中了\n";
        std::cout << "愿君好运\n";
        return;
    }
}

// ====================================
int main(int argc, const char **argv)
{
    // 设置随机数种子, 并生成一个随机数
    std::srand(time(0));
    int randomNumber = getRand(0, 100);

    std::cout << "============ 猜测随机数[0~100] ============\n";
    guessRandomNumber(randomNumber);

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\08_logicBranch.cpp -std=c++23
// g++ .\08_logicBranch.cpp -std=c++23
