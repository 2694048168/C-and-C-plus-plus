/**
 * @file 06_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++代码组织方式之函数
 * @version 0.1
 * @date 2024-03-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief C++中函数三要素: 返回值 + 函数名 + 参数列表
 * 打印所需要显示的消息;
 * 
 * @param message
 */
void printMessage(const std::string &message)
{
    std::cout << message << std::endl;
}

/**
 * @brief C++中函数三要素: 返回值 + 函数名 + 参数列表
 * 计算两个浮点数相加后的结果;
 * 
 * @param num1 
 * @param num2 
 * @return float 
 */
float addNumber(const float num1, const float num2)
{
    return num1 + num2;
}

/**
 * @brief C++中函数三要素: 返回值 + 函数名 + 参数列表
 * 计算两个浮点数相乘后的结果;
 * 
 * @param num1 
 * @param num2 
 * @return float 
 */
float mulNumber(const float num1, const float num2)
{
    return num1 * num2;
}

// ===================================
int main(int argc, const char **argv)
{
    printMessage("======== two number add ========");
    float sum = addNumber(2.4f, 12.f);
    std::cout << '\t' << sum << std::endl;

    printMessage("======== two number mul ========");
    float val1 = 3.21;
    float val2 = 2.1f;
    float mul  = mulNumber(val1, val2);
    std::cout << '\t' << mul << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\06_function.cpp -std=c++23
// g++ .\06_function.cpp -std=c++23
