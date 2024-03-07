/**
 * @file 03_dataTypeNumeric.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中基本的数据类型之数值型
 * @version 0.1
 * @date 2024-03-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// ===================================
int main(int argc, const char **argv)
{
    // 整数类型
    std::cout << "============ 整数类型 ============\n";
    int       valSignedInt   = -42;
    unsigned  valUnsignedInt = 42;
    short     valShort       = 12;
    long      valLong        = 999999;
    long long valLongLong    = 9999999999;

    std::cout << "the 'int' type sizeof: " << sizeof(valSignedInt);
    std::cout << " and value is " << valSignedInt << '\n';

    std::cout << "the 'unsigned int' type sizeof: " << sizeof(unsigned int) << '\n';
    std::cout << "the 'unsigned' type sizeof: " << sizeof(valUnsignedInt);
    std::cout << " and value is " << valUnsignedInt << '\n';

    std::cout << "the 'short' type sizeof: " << sizeof(valShort);
    std::cout << " and value is " << valShort << '\n';

    std::cout << "the 'long' type sizeof: " << sizeof(long);
    std::cout << " and value is " << valLong << '\n';

    std::cout << "the 'long long' type sizeof: " << sizeof(valLongLong);
    std::cout << " and value is " << valLongLong << '\n';

    // 浮点数类型
    std::cout << "============ 浮点数类型 ============\n";
    float  valFloat  = 3.14f;
    double valDouble = 3.14;
    std::cout << "the 'float' type sizeof: " << sizeof(valFloat);
    std::cout << " and value is " << valFloat << '\n';

    std::cout << "the 'double' type sizeof: " << sizeof(double);
    std::cout << " and value is " << valDouble << '\n';

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\03_dataTypeNumeric.cpp -std=c++23
// g++ .\03_dataTypeNumeric.cpp -std=c++23
