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

#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

/* Amicable numbers
Write a program that prints the list of all pairs of amicable numbers
 smaller than 1,000,000.
--------------------------------------- */

/* Solution
Two numbers are said to be amicable if the sum of the proper divisors of one number
 is equal to that of the other number. 

The proper divisors of a number are the positive prime factors of
 the number other than the number itself. 
Amicable numbers should not be confused with friendly numbers. 

相亲数，又称亲和数、友爱数、友好数，
指两个正整数中，彼此的全部正约数之和与另一方相等。
毕达哥拉斯曾说：“朋友是你灵魂的倩影，要像220与284一样亲密。”
每一对亲和数都是过剩数配亏数，较小的是过剩数，较大的是亏数。
------------------------------------------------ */

/**
 * @brief To determine the sum of proper divisors, 
 try all numbers from 2 to the square root of the number 
 (all prime factors are less than or equal to this value). 
If the current number, let’s call it i, divides the number, 
then i and num/i are both divisors. However, 
 if they are equal (for example, if i = 3, and n = 9, then i divides 9, but n/i = 3),
 we add only i because proper divisors must only be added once.
Otherwise, we add both i and num/i and continue:
------------------------------------------------- */
int sum_proper_divisors(const int number)
{
    int result = 1;
    for (int i = 2; i <= std::sqrt(number); ++i)
    {
        if (number % i == 0)
        {
            result += (i == (number / i)) ? i : (i + number / i);
        }
    }

    return result;
}

/**
 * @brief The solution to this problem is to iterate through all the numbers 
 up to the given limit. For each number, compute the sum of its proper divisors.
 Let’s call this sum1. 
 Repeat the process and compute the sum of the proper divisors of sum1.
 If the result is equal to the original number, 
 then the number and sum1 are amicable numbers:
 * 
 * @param limit 
 */
void print_amicable(const int limit)
{
    for (int number = 4; number < limit; ++number)
    {
        auto sum1 = sum_proper_divisors(number);

        if (sum1 < limit)
        {
            auto sum2 = sum_proper_divisors(sum1);

            if (sum2 == number && number != sum1)
            {
                std::cout << "(" << number << "," << sum1 << ") is Amicable number\n";
            }
        }
    }
}

// The above function prints pairs of numbers twice,
// such as 220,284 and 284,220.
// Modify this implementation to only print each pair a single time.
void print_amicable_once(const int limit)
{
    std::set<int> printed;
    for (int number = 4; number < limit; ++number)
    {
        if (printed.find(number) != printed.end())
            continue;

        auto sum1 = sum_proper_divisors(number);

        if (sum1 < limit)
        {
            auto sum2 = sum_proper_divisors(sum1);

            if (sum2 == number && number != sum1)
            {
                printed.insert(number);
                printed.insert(sum1);

                std::cout << "(" << number << "," << sum1 << ") is Amicable number\n";
            }
        }
    }
}

// -----------------------------
int main(int argc, char **argv)
{
    const unsigned long long LIMIT = 1000000;

    auto start = std::chrono::high_resolution_clock::now();

    // print_amicable(LIMIT);
    print_amicable_once(LIMIT);

    auto end = std::chrono::high_resolution_clock::now();

    auto time_consumption = end - start; /* the unit is 'ns' */
    std::cout << "========================================\n";
    std::cout << "the time consumption: " << time_consumption.count() << " ns\n";
    std::cout << "the time consumption: " << time_consumption.count() / 10e6 << " ms\n";
    std::cout << "the time consumption: " << time_consumption.count() / 10e9 << " s\n";
    std::cout << "========================================\n";

    return 0;
}
