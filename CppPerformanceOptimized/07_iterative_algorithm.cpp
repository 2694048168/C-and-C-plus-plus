/**
 * @file 07_iterative_algorithm.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

#include <iostream>

// 判断一个整数是否是 2 的幂的迭代算法的一种实现
inline bool is_power_2_iterative(unsigned n)
{
    for (unsigned one_bits = 0; n != 0; n >>= 1)
        if ((n & 1) == 1)
            if (one_bits != 0)
                return false;
            else
                one_bits += 1;
    return true;
}

// 判断一个整数是否是 2 的幂的闭形式
inline bool is_power_2_closed(unsigned n)
{
    return ((n != 0) && !(n & (n - 1)));
}

// -----------------------------------
int main(int argc, const char *argv[])
{
    constexpr unsigned NUM          = 10120122113;
    constexpr unsigned NUM_REPEATED = 1000000000000;

    bool flag_iter   = true;
    bool flag_closed = true;

    {
        Stopwatch{"====== Iter-algorithm ======"};
        for (size_t iter{0}; iter < NUM_REPEATED; ++iter)
        {
            is_power_2_iterative(NUM);
        }
    }

    {
        Stopwatch{"====== Closed-way ======"};
        for (size_t iter{0}; iter < NUM_REPEATED; ++iter)
        {
            is_power_2_closed(NUM);
        }
    }

    std::cout << std::boolalpha << flag_iter << '\n';
    std::cout << std::boolalpha << flag_closed << '\n';

    return 0;
}
