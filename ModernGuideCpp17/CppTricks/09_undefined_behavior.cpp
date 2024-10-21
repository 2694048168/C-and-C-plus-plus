/**
 * @file 09_undefined_behavior.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief  C/C++ 中的未定义行为(Undefined Behavior, UB)
 * @version 0.1
 * @date 2024-10-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief C/C++ 中的未定义行为(Undefined Behavior, UB)
 * *1. 实现定义行为(C/C++ standard ISO 标准):
 * 实现定义行为是指程序的行为依赖于具体的实现,而标准要求实现必须为每种行为提供文档说明.
 * 例如, int 类型在不同环境下的大小可能不同, 标准要求至少为 16 位, 而大多数环境下为 32 位.
 *
 * *2. 未指明行为(依赖于具体编译器的实现 GCC/Clang/MSVC/Inter C++/...(编译器扩展)
 * 未指明行为也是依赖于具体实现, 但标准并不要求提供文档说明.
 * 虽然行为可能变化, 但其结果应是合法的; 比如, 变量的分配方式和位置可以是连续的, 也可以是分开的.
 * 
 * *3. 未定义行为 UB
 * 未定义行为则是对程序行为没有任何限制, 标准不要求程序产生任何合法或有意义的结果.
 * 例如 访问非法内存就是未定义行为.
 * 
 * ?为什么会有未定义行为?
 * C/C++ 的设计目标之一是高效, 因此未定义行为的存在使得编译器能够优化程序.
 * 检测未定义行为的难度较大, 例如 带符号整数溢出并不总是会在编译阶段显现出来;
 * 若编译器必须处理这些未定义行为, 可能会影响程序的优化能力.
 * 因此 将某些操作定义为未定义行为, 编译器可以在优化时忽略这些情况, 从而生成更高效的代码;
 * !这也是为何在开启优化选项后, 程序可能会表现出意料之外的行为.
 * 
 * ?未定义行为的例子
 * 1. 带符号整数算术溢出;
 * 2. 越界访问;
 * 3. 无可视副作用的无限循环;
 * 4. 无法确定的运算顺序;
 * 5. 访问未初始化变量;
 * 
 * ?如何检测未定义行为?
 * *虽然编译期检测未定义行为较为困难, 但运行时可以通过一些工具来捕捉.
 * 1. 使用 -fsanitize=undefined;
 * 2. 使用 Valgrind 强大的内存调试工具, 可以帮助检测内存错误, 包括未定义行为;
 * 3. 使用 AddressSanitizer;运行时检测工具，专门用于检测内存错误和未定义行为;
 * 4. 在 CLion/Visual Studio 下使用 Google Sanitizers;
 * 5. CMakeLists.txt
if (ENABLE_ASAN)
    message(STATUS "build with ASAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif ()
 * 6. 使用 Clang Static Analyzer;
 * 7. 代码缺陷静态检查工具 Cppcheck;
 * 
 * *未定义行为(UB)是 C/C++ 的一大特色, 虽然它可以带来性能优化, 但也增加了程序调试的复杂性.
 * *避免未定义行为的关键在于养成良好的编程习惯, 并利用工具进行检测.
 * *通过理解未定义行为的本质, 程序员可以更有效地编写健壮、可预测的代码.
 * 
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    // 1. 带符号整数算术溢出

    // int 有符号数的范围是 [-2147483648, 2147483647]
    // unsigned int 无符号数的范围是 [0, 4294967295]
    int x = 2147483647;
    std::cout << x + 1 << std::endl;
    // 检查溢出
    if (x > 0 && x + 1 < x)
    {
        std::cout << "Overflow!\n";
    }
    else
    {
        std::cout << "Not overflow!\n";
    }
    /* 在开启优化选项时, 可能会发现预期的 "Overflow!" 并未出现;
      原因在于带符号整数溢出被视为未定义行为, 编译器因此可能忽略了该情况.
    g++ 09_undefined_behavior.cpp
    g++ 09_undefined_behavior.cpp -g
    g++ -fsanitize=undefined -o undefined_behavior 09_undefined_behavior.cpp
    */

    // 2. 越界访问
    int arr[5] = {0, 1, 2, 3, 4};
    int index  = 5;
    // 检查越界
    if (index >= 0 && index < 5)
    {
        std::cout << "数组中的值: " << arr[index];
    }
    else
    {
        std::cout << "数组中的值: " << arr[index];
        std::cout << "\n索引越界!\n";
    }
    /* C/C++ 并不自动进行数组越界检查, 导致可能出现以下后果:
    1. 访问非法内存引发运行时错误runtime-error(RE);
    2. 意外修改其他变量的值;
    * 不进行越界检查的原因在于其成本较高,并可能影响程序的优化机会.
    g++ 09_undefined_behavior.cpp
    g++ 09_undefined_behavior.cpp -O2
    g++ 09_undefined_behavior.cpp -g
    */

    // 3. 无可视副作用的无限循环
    auto checkCondition = []() -> bool
    {
        unsigned cnt = 0;
        while (true)
        {
            if (cnt < 0)
                return true; // 这个条件永远不会为真
        }
        return false;
    };
    // if (checkCondition())
    if (false)
    {
        std::cout << "This program has been terminated.\n";
    }
    else
    {
        std::cout << "Some strange things happened!\n";
    }
    /* 由于 checkCondition() 函数中的无限循环为未定义行为,
     编译器可能(GCC/Clang/MSVC/Inter C++)会将其优化掉, 从而导致不同的行为表现.
    clang++ 09_undefined_behavior.cpp
    clang++ 09_undefined_behavior.cpp -O2
    clang++ 09_undefined_behavior.cpp -g
     */

    // 4. 无法确定的运算顺序
    int num    = 1;
    int result = (num++ + ++num); // 无法确定的运算顺序
    std::cout << "结果: " << result << std::endl;
    /* num++ 和 ++num 的副作用无顺序, 因此结果是未定义的.
     clang++ 09_undefined_behavior.cpp -O2
     */

    // 5. 访问未初始化变量
    int val;                                               // 未初始化
    std::cout << "未初始化变量的值: " << val << std::endl; // 结果未定义
    /* 访问未初始化的变量同样是未定义行为, 可能导致不确定的输出.
     clang++ 09_undefined_behavior.cpp -O2
     g++ 09_undefined_behavior.cpp -O2
     */

    return 0;
}
