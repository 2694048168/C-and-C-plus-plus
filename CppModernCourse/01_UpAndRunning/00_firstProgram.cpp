/**
 * @file 00_firstProgram.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <iostream>

int step_function(int x)
{
    int result = 0;
    if (x < 0)
    {
        result = -1;
    }
    else if (x > 0)
    {
        result = 1;
    }
    return result;
}

/**
 * @brief Main: A C++ Program’s Starting Point
 * C++ 程序的入口点
 * 
 * Libraries: Pulling in External Code 引入外部代码
 * Libraries are helpful code collections you can import into your programs
 * to prevent having to reinvent the wheel. 
 * ---- Python, Go, and Java have import.
 * ---- Rust, PHP, and C# have use/using.
 * ---- JavaScript, Lua, R, and Perl have require/requires.
 * ---- C and C++ have #include.
 * 
 * 编译工具链: 预处理器(preprocessor) --> 编译器(compiler) --> 链接器(linker)
 * 1. source.cpp ---> 编译单元 source_unit.cpp
 * 2. 编译单元 source_unit.cpp ---> 目标文件 source.obj
 * 3. 目标文件 source.obj ---> 链接生成最终可执行文件(*.exe or *.lib)
 * 
 * 开发环境(IDE): 代码编辑器 + 编译工具链 + 调试器
 * 1. Windows OS ---> Visual Studio
 * 2. macOS ---> Xcode
 * 3. Linux or Unix OS ---> GCC
 * 
 * C++是一门面向对象的语言, 对象是关于状态和行为的抽象,
 * 行为和状态的集合被用来描述对象, 称之为 类型, 
 * C++是一门强类型语言, 这这意味着每一个对象都有一个预先定义好的数据类型,
 * C++有一个内建整数类型(built-in), 即 int,
 * int 对象可以存储整数(状态), 并且也支持许多数学运算(行为),
 * 要使用 int 类型来做一些有意义的任务, 需要创建一些 int 对象, 并且对它们命名, 命名对象被称为变量
 * 
 * 
 * @param argc 
 * @param argv 
 * @return int 程序返回退出码(操作系统判断是否正常运行)给到操作系统, 并结束运行
 *
 */
// -----------------------------------
int main(int argc, const char **argv)
{
    printf("Helle C++ world!\n");

    // C++是一门强类型语言, 这这意味着每一个对象都有一个预先定义好的数据类型
    int val = 42;
    int num = 12;
    std::cout << val << " + " << num << " = " << val + num << '\n';
    /**
     * @brief The C++ Type System
     * step 1. Declaring Variables 声明变量
     * step 2. Initializing a Variable’s State 初始化变量状态
     * 
     * Conditional Statements and Boolean expressions 条件语句和布尔表达式
     * 
     */
    int x = 0;
    if (x > 0)
        printf("Positive value: %d", x);
    else if (x < 0)
        printf("Negative value: %d", x);
    else
        printf("Zero value: %d", x);

    /**
     * @brief Function 函数
     * 函数是一种代码块, 接受任意数量的输入对象(参数列表), 同时将输出对象返回给到调用者.
     * return_type function_name(par_type1 par_name1, par_type2 par_name12)
     * {
            return return_type;
     * }
     * 
     * A Step Function 阶跃函数
     * 调用函数, 需要使用函数的名称、大括号，以及一系列逗号隔开的必需参数,
     * 编译器会从头到尾读取文件内容, 所以函数的声明必须出现在它第一次被使用之前.
     * 
     */
    std::cout << "the result of Step Function: " << step_function(42) << '\n';
    std::cout << "the result of Step Function: " << step_function(-99) << '\n';
    std::cout << "the result of Step Function: " << step_function(x) << '\n';

    /**
     * @brief printf格式指定符
     * 打印字符串常量, printf 还可以将多个值组成一个格式良好的字符串,
     *  是一种特殊的函数, 可以接受一个或者多个参数;
     * printf 的第一个参数一直是格式化字符串, 格式化字符串为要打印的字符串提供模板,
     * 并且它可以包含任意数量的特殊格式指定符(format specifier),所有格式指定符都以 %开头
     * 格式指定符告诉 printf 如何解释和格式化跟在格式化字符串后面的参数.
     * 
     */
    printf("Ten %d, Twenty %d, Thirty %d\n", 10, 20, 30);

    /**
     * @brief iostream、printf和输入/输出教学法
     * C++新手哪种标准输出方法有非常强烈的意见: 一种方法是 printf, 它的血统可以追溯到C语言
     * 另一种方法是 std::cout, 它是C+ +标准库的 iostream 库的一部分;
     * 1. 什么是流缓冲区?
     * 2. 什么是 operator<<?
     * 3. 什么是方法? 
     * 4. flush() 是如何工作的?
     * 5. std::cout 会在析构函数中自动刷新吗?
     * 6. 什么是析构函数? 
     * 7. setf 是什么?
     * 8. 格式化标志又是什么?
     * 9. 是 BitmaskType吗?
     * 10. 什么是操纵符?
     * printf 也有问题, 一旦你学会了 std::cout, 应该会更喜欢它
     * 使用 printf 很容易导致格式指定符和参数不匹配, 进而导致奇怪的行为, 
     * 还可能导致程序崩溃, 甚至出现安全漏洞. 
     * 使用 std::cout意味着不需要格式化字符串了, 所以也就不需要记住格式指定符了,
     * iostream 也是可扩展的, 这意味着可以将输入和输出功能集成到自己的类型中
     *
     */

    // ! Debugging
    // * One of the most important skills for a software engineer is efficient, effective debugging.
    /* 软件工程师最重要的技能之一是高效、有效的调试能力. */

    /**
     * @brief Debugging
     * 1. Windows 操作系统建议使用 Visual Studio IDE进行调试;
     * 2. macOS or Linux 操作系统建议使用 CLion IDE进行调试;
     * 3. 跨平台操作系统, 建议使用 VSCode + CMake + GDB/LLDB/VS调试器
     */

    return 0;
}
