/**
 * @file 11_passParam.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习之函数参数的三种传递方式
 * @version 0.1
 * @date 2024-03-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

inline void passByValue(double timeComsume)
{
    timeComsume += 2.0;
}

inline void passByReference(double &timeComsume)
{
    timeComsume += 2.0;
}

inline void passByPointer(double *pTimeComsume)
{
    *pTimeComsume += 2.0;
}

// ====================================
int main(int argc, const char **argv)
{
    double value = 3.12;
    std::cout << "[INFO] 初始数值: " << value << std::endl;

    std::cout << "========= pass by value(copy) =========\n";
    passByValue(value);
    std::cout << "[INFO] 值传递后数值: " << value << std::endl;

    std::cout << "========= pass by reference(address) =========\n";
    passByReference(value);
    std::cout << "[INFO] 引用传递后数值: " << value << std::endl;

    std::cout << "========= pass by pointer(address) =========\n";
    passByPointer(&value);
    std::cout << "[INFO] 指针传递后数值: " << value << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\11_passParam.cpp -std=c++23
// g++ .\11_passParam.cpp -std=c++23
