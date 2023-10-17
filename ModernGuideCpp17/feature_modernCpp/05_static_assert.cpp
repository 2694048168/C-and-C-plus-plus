/**
 * @file 05_static_assert.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cassert> /* assert 宏 */
#include <cstring>
#include <iostream>

// 创建一个指定大小的 char 类型数组
// 此处 const 修饰含义表明为 "只读", 并不能表达 "常量"含义
char *createArray(const int size)
{
    // 通过断言判断数组大小是否大于0
    // assertion 也是一种减少程序嵌套判断的手段，
    // assert是一个运行时断言，也就是说它只有在程序运行时才能起作用
    assert(size > 0); /* 必须大于0, 否则程序中断 */

    char *array = new char[size];

    return array;
}

// 断言 assertion 是一种编程中常用的手段
// 断言就是将一个返回值总是需要为真的判断表达式放在语句中,
// 用于排除在设计的逻辑上不应该产生的情况
// ------------------------------
// 静态断言 static_assert, 所谓静态就是在编译时就能够进行检查的断言, 使用时不需要引用头文件
// 静态断言的另一个好处是, 可以自定义违反断言时的错误提示信息。
// 静态断言使用起来非常简单，它接收两个参数：
// 参数1：断言表达式，这个表达式通常需要返回一个 bool值
// 参数2：警告信息，它通常就是一段字符串，在违反断言（表达式为false）时提示该信息
// ------------------------------
int main(int argc, char **argv)
{
    constexpr int SIZE = 20;
    // constexpr int SIZE = 0;

    char *buf = createArray(SIZE);

    strcpy_s(buf, 16, "hello, world!");
    std::cout << "buf = " << buf << std::endl;

    delete[] buf;

    // ======= static assert in C++11 =======
    // static_assert(sizeof(char) == 4, "Error, Not is 32bit platform...");
    static_assert(sizeof(long) == 4, "Error, Not is 32bit platform...");
    // 静态断言的表达式是在编译阶段进行检测, 所以这个表达式必须是常量表达式

    std::cout << "64 bit platform pointer size: " << sizeof(char *) << std::endl;
    std::cout << "64 bit platform long size: " << sizeof(long) << std::endl;

    return 0;
}