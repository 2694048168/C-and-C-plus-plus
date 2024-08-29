/**
 * @file 01_long_integer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** C++11 标准要求 long long 整型可以在不同平台上有不同的长度，但至少有64位
 * long long - 对应类型的数值可以使用 LL (大写) 或者 ll (小写) 后缀;
 * unsigned long long - 对应类型的数值可以使用 ULL (大写) 或者 ull (小写) 或者 Ull、uLL (等大小写混合)后缀;
 * 
 * ! 扩展的整形, 在C++11中一共只定义了以下5种标准的有符号整型：
 * signed char -- short int -- int -- long int -- long long int
 * 标准同时规定，每一种有符号整型都有一种对应的无符号整数版本,
 * 且有符号整型与其对应的无符号整型具有相同的存储空间大小. 
 * 比如与 signed int 对应的无符号版本的整型是 unsigned int.
 * !在C++中处理数据的时候，如果参与运算的数据或者传递的参数类型不匹配，
 * !整型间会发生隐式的转换，这种过程通常被称为整型的提升.
 * 关于这种整形提升的隐式转换遵循如下原则:
 * *1. 长度越大的整型等级越高,比如 long long int 的等级会高于int.
 * *2. 长度相同的情况下,标准整型的等级高于扩展类型,比如 long long int 和 int64 
 *     如果都是64 位长度，则long long int类型的等级更高.
 * *3. 相同大小的有符号类型和无符号类型的等级相同, long long int 和unsigned longlong int的等级就相同.
 * *4. 转换过程中, 低等级整型需要转换为高等级整型, 有符号的需要转换为无符号整形.
 * 
 */

#include <iostream>

// --------------------------------
int main(int argc, char **argv)
{
    long long num1 = 123456789LL;
    long long num2 = 123456789ll;

    std::cout << "the size of bytes: " << sizeof(num1) << " and value: " << num1
              << " and the type: " << typeid(num1).name() << '\n';
    std::cout << "the size of bytes: " << sizeof(long long) << " and value: " << num1
              << " and the type: " << typeid(long long).name() << '\n';

    unsigned long long num5 = 123456789ULL;
    unsigned long long num6 = 123456789ull;
    unsigned long long num3 = 123456789uLL;
    unsigned long long num4 = 123456789Ull;
    std::cout << "the size of bytes: " << sizeof(num5) << " and value: " << num5
              << " and the type: " << typeid(num5).name() << '\n';
    std::cout << "the size of bytes: " << sizeof(unsigned long long) << " and value: " << num6
              << " and the type: " << typeid(unsigned long long).name() << '\n';

    /** 事实上在C++11中还有一些类型与以上两种类型是等价的:
     * 对于有符号类型的 long long和以下三种类型等价
     * long long int --- signed long long --- signed long long int
     * 对于无符号类型的unsigned long long 和unsigned long long int是等价的 */
    // 同其他的整型一样, 要了解平台上 long long大小的方法就是查看<climits>中的宏
    // 这个值根据平台不同会有所变化，原因是因为C++11标准规定该类型至少占8字节，
    // 它占的字节数越多，对应能够存储的数值也就越大。
    long long          max    = LLONG_MAX;
    long long          min    = LLONG_MIN;
    unsigned long long ullMax = ULLONG_MAX;

    std::cout << "Max Long Long value: " << max << std::endl
              << "Min Long Long value: " << min << std::endl
              << "Max unsigned Long Long value: " << ullMax << std::endl;

    return 0;
}
