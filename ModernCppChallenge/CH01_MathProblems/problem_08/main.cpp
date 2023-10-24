/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

/* Armstrong numbers
Write a program that prints all Armstrong numbers with three digits.
--------------------------------------- */

/* Solution
An Armstrong number (named so after Michael F. Armstrong), 
also called a narcissistic number, a pluperfect digital invariant, 
or a plus perfect number, is a number that is equal to the sum of its own digits
 when they are raised to the power of the number of digits.
As an example, the smallest Armstrong number is 153, 
which is equal to 1^3 + 5^3 + 3^3 = 153. 
------------------------------------------------ */

/* To determine if a number with three digits is a narcissistic number,
you must first determine its digits in order to sum their powers. 
However, this involves division and modulo operations, which are expensive.
 A much faster way to compute it is to rely on the fact that a number is
  a sum of digits multiplied by 10 at the power of their zero-based position.
In other words, for numbers up to 1,000, we have a*10^2 + b*10^2 + c. 
Since you are only supposed to determine numbers with three digits, 
that means a would start from 1. 
This would be faster than other approaches because multiplications are faster to
compute than divisions and modulo operations. 
An implementation of such a function would look like this:
----------------------------------------------------------- */
template<typename Time = std::chrono::microseconds, typename Clock = std::chrono::high_resolution_clock>
struct perf_timer
{
    template<typename F, typename... Args>
    static Time duration(F &&f, Args... args)
    {
        auto start = Clock::now();

        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

        auto end = Clock::now();

        return std::chrono::duration_cast<Time>(end - start);
    }
};

void print_narcissistic_1(const bool printResults)
{
    for (int a = 1; a <= 9; a++)
    {
        for (int b = 0; b <= 9; b++)
        {
            for (int c = 0; c <= 9; c++)
            {
                auto abc = a * 100 + b * 10 + c;
                auto arm = a * a * a + b * b * b + c * c * c;
                if (abc == arm)
                {
                    if (printResults)
                        std::cout << arm << '\n';
                }
            }
        }
    }

    // std::cout << "========================" << std::endl;
}

void print_narcissistic_2(const bool printResults)
{
    for (int i = 100; i <= 1000; ++i)
    {
        int arm = 0;
        int n   = i;
        while (n > 0)
        {
            auto d = n % 10;
            n      = n / 10;
            arm += d * d * d;
        }

        if (i == arm)
        {
            if (printResults)
                std::cout << arm << '\n';
        }
    }

    // std::cout << "========================" << std::endl;
}

void print_narcissistic_3(const int limit, const bool printResults)
{
    for (int i = 1; i <= limit; ++i)
    {
        std::vector<int> digits;
        int              n = i;
        while (n > 0)
        {
            digits.push_back(n % 10);
            n = n / 10;
        }

        int arm = std::accumulate(std::begin(digits), std::end(digits), 0,
                                  [s = digits.size()](const int sum, const int digit)
                                  { return sum + static_cast<int>(std::pow(digit, s)); });

        if (i == arm)
        {
            if (printResults)
                std::cout << arm << '\n';
        }
    }

    // std::cout << "========================" << std::endl;
}

// -----------------------------
int main(int argc, char **argv)
{
    print_narcissistic_1(true);
    print_narcissistic_2(true);
    print_narcissistic_3(1000, true);

    std::cout << "=====================================\n";
    // 测试耗时
    auto timer1 = perf_timer<>::duration(
        []()
        {
            for (int i = 0; i < 10000; ++i)
            {
                print_narcissistic_1(false);
            }
        });

    std::cout << std::chrono::duration<double, std::milli>(timer1).count() << " ms\n";

    auto timer2 = perf_timer<>::duration(
        []()
        {
            for (int i = 0; i < 10000; ++i)
            {
                print_narcissistic_2(false);
            }
        });

    std::cout << std::chrono::duration<double, std::milli>(timer2).count() << " ms\n";

    auto timer3 = perf_timer<>::duration(
        []()
        {
            for (int i = 0; i < 10000; ++i)
            {
                print_narcissistic_3(1000, false);
            }
        });

    std::cout << std::chrono::duration<double, std::milli>(timer3).count() << " ms\n";

    return 0;
}
