/**
 * @file 06_static_assert.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 断言（assertion）是一种编程中常用的手段, 
 * 在通常情况下, 断言就是将一个返回值总是需要为真的判断表达式放在语句中,
 * 用于排除在设计的逻辑上不应该产生的情况. 比如：一个函数总需要输入在一定的范围内的参数,
 * 那么就可以对该参数使用断言,以迫使在该参数发生异常的时候程序退出,从而避免程序陷入逻辑的混乱.
 * 从一些意义上讲, 断言并不是正常程序所必需的, 
 * 不过对于程序调试来说, 通常断言能够帮助程序开发者快速定位那些违反了某些前提条件的程序错误.
 * 如果我们要在C++程序中使用断言, 需要在程序中包含头文件<cassert>头文件中
 * 提供 assert 宏, 用于在运行时进行断言.
 * 
 * 2. 静态断言
 * assert是一个运行时断言,也就是说它只有在程序运行时才能起作用.
 * 这意味着不运行程序将无法得知某些条件是否是成立的.
 * 比如想知道当前是32位还是64位平台, 对于这个需求应该是在程序运行之前就应该得到结果,
 * 如果使用断言显然是无法做到的, 对于这种情况就需要使用C++11提供的静态断言了.
 * 静态断言static_assert, 所谓静态就是在编译时就能够进行检查的断言, 
 * 使用时不需要引用头文件. 静态断言的另一个好处是,可以自定义违反断言时的错误提示信息.
 * 静态断言使用起来非常简单，它接收两个参数：
 * ?参数1：断言表达式，这个表达式通常需要返回一个 bool值
 * ?参数2：警告信息，它通常就是一段字符串，在违反断言（表达式为false）时提示该信息
 * 
 */

#include <cassert>
#include <cstring>
#include <iostream>

// 创建一个指定大小的 char 类型数组
char *createArray(int size)
{
    // 通过断言判断数组大小是否大于0
    // 使用了断言assert(expression) ,这是一个宏, 它的参数是一个表达式
    // 表达式通常返回一个布尔类型的值, 要求表达式必须为 true 程序才能继续向下执行,否则会直接中断.
    assert(size > 0); // 必须大于0, 否则程序中断
    char *array = new char[size];
    return array;
}

// -------------------------------------
int main(int argc, const char **argv)
{
    // char *buf = createArray(0);
    char *buf = createArray(20);
    // 此处使用的是vs提供的安全函数, 也可以使用 strcpy
    strcpy_s(buf, 16, "hello, world!");
    // std::strcpy(buf, "hello, world!");
    std::cout << "buf = " << buf << std::endl;
    delete[] buf;

    // ==============================
    // !由于静态断言的表达式是在编译阶段进行检测,
    // !所以在它的表达式中不能出现变量，也就是说这个表达式必须是常量表达式
    static_assert(sizeof(long) == 4, "错误, 不是32位平台...");
    std::cout << "64bit Linux 指针大小: " << sizeof(char *) << std::endl;
    std::cout << "64bit Linux long 大小: " << sizeof(long) << std::endl;

    // 32位系统与64位系统各数据类型对比
    // *	地址（指针）4	8
    // long	长整型	   4   8

    return 0;
}
