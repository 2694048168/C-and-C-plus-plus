/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <numeric>
#include <vector>

/* Least common multiple 最小公倍数
Write a program that will, given two or more positive integers,
calculate and print the least common multiple of them all.
------------------------------------------------------------ */

/* Solution
Algorithm to Find LCM of Two Numbers:
- LCM of two integers a and b is the smallest positive integer
 that is divisible by both a and b.

The product of the Least Common Multiple(LCM) and Greatest Common Divisor(GCD)
 of two numbers a and b is equal to the product of numbers a and b itself.
  such as, a X b = LCM(a, b) * GCD(a, b)
  then ===> LCM(a, b) = (a X b) / GCD(a, b)
---------------------------------------------- */

int gcd_func(const int a, const int b)
{
    return b == 0 ? a : std::gcd(b, a % b);
}

int lcm_func(const int a, const int b)
{
    int h = gcd_func(a, b);

    return h ? (a * (b / h)) : 0;
}

// two or more positive integers
// template<typename InputIt>
template<class InputIt>
int lcm_recursive(InputIt first, InputIt last)
{
    return std::accumulate(first, last, 1, lcm_func);
}

// the second way by using the std::lcm since C++17
#define OUT(...) std::cout << #__VA_ARGS__ << " = " << __VA_ARGS__ << '\n'

// since C++20 supported
constexpr auto lcm(auto x, auto... xs)
{
    return ((x = std::lcm(x, xs)), ...);
}

// -----------------------------
int main(int argc, char **argv)
{
    int num = 0;
    std::cout << "Please Input the number of count: ";
    std::cin >> num;

    std::vector<int> numbers;
    std::cout << "Please Input the value of these numbers: ";
    for (int i = 0; i < num; ++i)
    {
        int v{0};
        std::cin >> v;
        numbers.push_back(v);
    }

    std::cout << "lcm = " << lcm_recursive(std::begin(numbers), std::end(numbers)) << std::endl;

    // ======= second way by using std::lcm ========
    std::cout << "================================\n";
    constexpr int p{2 * 2 * 3};
    constexpr int q{2 * 3 * 3};
    static_assert(2 * 2 * 3 * 3 == std::lcm(p, q));
    static_assert(225 == std::lcm(45, 75));

    static_assert(std::lcm(6, 10) == 30);
    static_assert(std::lcm(6, -10) == 30);
    static_assert(std::lcm(-6, -10) == 30);

    static_assert(std::lcm(24, 0) == 0);
    static_assert(std::lcm(-24, 0) == 0);

    OUT(lcm(2 * 3, 3 * 4, 4 * 5));
    OUT(lcm(2 * 3 * 4, 3 * 4 * 5, 4 * 5 * 6));
    OUT(lcm(2 * 3 * 4, 3 * 4 * 5, 4 * 5 * 6, 5 * 6 * 7));

    return 0;
}
