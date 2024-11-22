/**
 * @file 03_string_optimizer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

#include <cassert>
#include <functional>
#include <memory>
#include <string>

/**
 * @brief 这个函数的功能是从一个由 ASCII 字符组成的字符串中移除控制字符.
 * 看起来它似乎很无辜, 但是出于多种原因, 这种写法的函数确实性能非常糟糕.
 * 这个函数是一个很好的例子, 展示了在编码时完全不考虑性能是多么地危险.
 * 
 * @param s 
 * @return std::string 
 */
std::string remove_ctrl(std::string s)
{
    std::string result;
    for (int i = 0; i < s.length(); ++i)
    {
        if (s[i] >= 0x20)
            result = result + s[i];
    }
    return result;
}

// 一种优化选择是尝试改进算法
// 一种优化选择是缓存参数字符串的长度，以减少外层 for 循环中结束条件语句的性能开销
std::string remove_ctrl_block(std::string s)
{
    std::string result;
    for (size_t b = 0, i = b, e = s.length(); b < e; b = i + 1)
    {
        for (i = b; i < e; ++i)
        {
            if (s[i] < 0x20)
                break;
        }
        result = result + s.substr(b, i - b);
    }
    return result;
}

// 选择一种更好的算法是一种多么强大的优化手段
std::string remove_ctrl_block_append(std::string s)
{
    std::string result;
    result.reserve(s.length());
    for (size_t b = 0, i = b; b < s.length(); b = i + 1)
    {
        for (i = b; i < s.length(); ++i)
        {
            if (s[i] < 0x20)
                break;
        }
        result.append(s, b, i - b);
    }
    return result;
}

// 使用 std::string 的 erase() 成员函数移除控制字符来改变字符串
std::string remove_ctrl_erase(std::string s)
{
    for (size_t i = 0; i < s.length();)
        if (s[i] < 0x20)
            s.erase(i, 1);
        else
            ++i;
    return s;
}

void test_driver(std::function<std::string(std::string)> func, std::string_view func_name, unsigned int multiplier)
{
    typedef unsigned long counter_t;

    counter_t iterations = 1000 * multiplier;

    std::string str("\07Now is the time\07 for all good men\r\n to come to the aid of their country. \07");
    std::string test("Now is the time for all good men to come to the aid of their country. ");
    str  = str + str + str;
    test = test + test + test;

    std::string result;
    {
        Stopwatch sw(func_name);
        for (counter_t idx = 0; idx < iterations; ++idx)
        {
            result = func(str);
        }
    }
    assert(result.compare(test) == 0);
}

// =====================================
int main(int argc, const char *argv[])
{
    const unsigned int repeater = 100;

    test_driver(remove_ctrl, "remove_ctrl()", repeater);
    test_driver(remove_ctrl_block, "remove_ctrl_block()", repeater);
    test_driver(remove_ctrl_block_append, "remove_ctrl_block_append()", repeater);
    test_driver(remove_ctrl_erase, "remove_ctrl_erase()", repeater);

    /* 
     * 1. 使用更好的编译器, 新版本的编译器可能会改善性能,不过这需要开发人员通过测试去验证,而不是想当然;
     * 2. 使用更好的字符串库, 使用std::stringstream避免值语义;std::string_view;
     * 如果 std::stringstream 是用 std::string 实现的,那么它在性能上永远不能胜过 std::string;
     * 它的优点在于可以防止某些降低程序性能的编程实践.
     * 3. 使用更好的内存分配器
     * 4. 消除字符串转换, 从以空字符结尾的字符串到 std::string 的无谓转换，是浪费计算机 CPU 周期;
     * 一个大型软件系统可能含有很多层（layer）,这会让字符串转换成为一个大问题.
     * 如果在某一层中接收的参数类型是 std::string, 而在它下面一层中接收的参数类型是 char*,
     * 那么可能需要写一些代码将 std::string 反转为 char*
     * 5. 不同字符集间的转换, 移除转换的最佳方法是为所有的字符串选择一种固定的格式，并将所有字符串都存储为这种格式;
     */

    return 0;
}
