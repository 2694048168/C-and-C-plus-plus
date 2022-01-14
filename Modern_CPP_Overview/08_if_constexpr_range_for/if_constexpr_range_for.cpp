/**
 * @file if_constexpr_range_for.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 将 constexpr 常量表达式用于 if 语句，在编译阶段进行优化，提升性能; range-based for loop
 * @version 0.1
 * @date 2022-01-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @brief C++11 introduces the constexpr keyword,
 * which compiles expressions or functions into constant results. 
 * A natural idea is that if we introduce this feature into the conditional judgment, 
 * let the code complete the branch judgment at compile-time, 
 * can it make the program more efficient? 
 * C++17 introduces the constexpr keyword into the if statement,
 * allowing you to declare the condition of a constant expression in your code.
 * 
 */
template <typename T>
auto print_type_info(const T &t)
{
    if constexpr (std::is_integral<T>::value)
    {
        return t + 1;
    }
    else
    {
        return t + 0.0015;
    }
}

int main(int argc, char **argv)
{
    std::cout << print_type_info(41) << std::endl;
    std::cout << print_type_info(3.14) << std::endl;

    // range-based for loop 范围 for 循环
    /* C++11 introduces a range-based iterative method, 
    and we can write loops that are as concise as Python, and we can further simplify */
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    if (auto itr = std::find(vec.begin(), vec.end(), 3); itr != vec.end())
    {
        *itr = 4;
    }

    for (auto element : vec)
    {
        std::cout << element << ' '; /* read only */
    }
    std::cout << "\n";
    
    for (auto &element : vec)
    {
        std::cout << element + 1 << ' '; /* writeable */
    }
    std::cout << "\n";

    return 0;
}
