/**
 * @file 05_loop_optimized.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

#include <iostream>

int main(int argc, const char **)
{
    char s[] = "This string has many space (0x20) chars. ";

    // 未优化的 for 循环
    {
        Stopwatch sw("string.strlen function");
        for (size_t idx = 0; idx < strlen(s); ++idx)
            if (s[idx] == ' ')
                s[idx] = '*';
    }

    // 缓存了循环结束条件值的 for 循环
    {
        Stopwatch sw("string.strlen len-condition cache");
        for (size_t idx = 0, len = strlen(s); idx < len; ++idx)
            if (s[idx] == ' ')
                s[idx] = '*';
    }

    // 将一个 for 循环简化为 do 循环通常可以提高循环处理的速度(现代编译器自动优化项)
    {
        Stopwatch sw("for into do-while");

        size_t i = 0, len = strlen(s); // for循环初始化表达式
        do
        {
            if (s[i] == ' ')
                s[i] = ' ';
            ++i; // for循环继续表达式
        }
        while (i < len); // for循环条件
    }

    // 对循环进行递减优化
    {
        Stopwatch sw("for decreased ");

        for (int i = (int)strlen(s) - 1; i >= 0; --i)
            if (s[i] == ' ')
                s[i] = '*';
    }

    return 0;
}
