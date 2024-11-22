/**
 * @file 02_string_optimizer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

#include <cassert>
#include <functional>
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

std::string remove_ctrl_mutating(std::string s)
{
    std::string result;
    for (int i = 0; i < s.length(); ++i)
    {
        if (s[i] >= 0x20)
            result += s[i];
        // 使用复合赋值操作避免临时字符串
        // 移除内存分配和复制操作来优化
        /* 这次改善源于移除了所有为了分配临时字符串对象来保存
        连接结果而对内存管理器的调用, 以及相关的复制和删除临时字符串的操作.
        赋值时的分配和复制操作也可以被移除, 不过这取决于字符串的实现方式.
        */
    }
    return result;
}

std::string remove_ctrl_reserve(std::string s)
{
    std::string result;
    result.reserve(s.length());
    // 通过预留存储空间减少内存的重新分配
    // 使用 reserve() 不仅移除了字符串缓冲区的重新分配,
    // 还改善了函数所读取的数据的缓存局部性(cache locality),
    // 因此从中得到了更好的改善效果
    for (int i = 0; i < s.length(); ++i)
    {
        if (s[i] >= 0x20)
            result += s[i];
    }
    return result;
}

// 移除实参复制, 省去了另外一次内存分配
// 由于内存分配是昂贵的, 所以哪怕只是一次内存分配, 也值得从程序中移除
std::string remove_ctrl_ref_args(const std::string &s)
{
    std::string result;
    result.reserve(s.length());
    for (int i = 0; i < s.length(); ++i)
    {
        if (s[i] >= 0x20)
            result += s[i];
    }
    return result;
}

std::string remove_ctrl_ref_args_it(const std::string &s)
{
    std::string result;
    result.reserve(s.length());
    // 使用迭代器消除指针解引, 可以节省两次解引操作
    for (auto it = s.begin(), end = s.end(); it != end; ++it)
    {
        if (*it >= 0x20)
            result += *it;
    }
    return result;
}

// 消除对返回的字符串的复制
// 明确编译器在省去调用复制构造函数时确实会进行的处理
void remove_ctrl_ref_result_it(std::string &result, const std::string &s)
{
    result.clear();
    result.reserve(s.length());
    for (auto it = s.begin(), end = s.end(); it != end; ++it)
    {
        if (*it >= 0x20)
            result += *it;
    }
}

// 用字符数组代替字符串
// 重写函数和改变它的接口, 可以获得很大的性能提升
void remove_ctrl_cstrings(char *pDest, const char *pSrc, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (pSrc[i] >= 0x20)
            *pDest++ = pSrc[i];
    }
    *pDest = 0;
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

void test_driver(std::function<void(std::string &, const std::string)> func, std::string_view func_name,
                 unsigned int multiplier)
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
            func(result, str);
        }
    }
    assert(result.compare(test) == 0);
}

// =====================================
int main(int argc, const char *argv[])
{
    const unsigned int repeater = 100;

    test_driver(remove_ctrl, "remove_ctrl()", repeater);
    test_driver(remove_ctrl_mutating, "remove_ctrl_mutating()", repeater);
    test_driver(remove_ctrl_reserve, "remove_ctrl_reserve()", repeater);
    test_driver(remove_ctrl_ref_args, "remove_ctrl_ref_args()", repeater);
    test_driver(remove_ctrl_ref_args_it, "remove_ctrl_ref_args_it()", repeater);
    test_driver(remove_ctrl_ref_result_it, "remove_ctrl_ref_result_it()", repeater);

    // !停下来思考, 我想我们可能走得太远了
    // ?在进行性能优化时, 要注意权衡简单性、安全性与所获得的性能提升效果
    // *对于一项性能改善是否值得增加接口的复杂性或是增加需要评审函数调用的工作量(Hot Spot Code)
    // 采取各种优化手段后的测试结果, 这些结果都来自于遵循一个简单的规则:
    // ?移除内存分配和相关的复制操作, 第一个优化手段带来的性能提升效果最显著.
    // 许多因素都会影响绝对时间, 包括处理器(CPU)、基础时钟频率(主频)、内存总线频率(频率)、
    // 编译器(compiler)和优化器
    // 调试版(Debug)和正式(Release)（优化后）版的测试结果来证明

    /* C/C++代码性能优化——编译器和CPU
    https://blog.csdn.net/feihe0755/article/details/136947654
     */

    return 0;
}
