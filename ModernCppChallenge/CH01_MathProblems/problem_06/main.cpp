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

#include <cmath>
#include <iostream>

/* Abundant numbers 过剩数 (Abundant number)
Write a program that prints all abundant numbers and their abundance, 
up to a number entered by the user.
--------------------------------------- */

/* Solution
In number theory, an abundant number or excessive number is a positive integer
 for which the sum of its proper divisors is greater than the number. 

在数论中，过剩数又称作丰数或盈数，
一般指的是真约数之和大于自身的一类正整数，
严格意义上指的是因数和函数大于两倍自身的一类正整数。

https://en.wikipedia.org/wiki/Abundant_number
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
 * @brief Printing abundant numbers is as simple as iterating up to the specified limit, 
 computing the sum of proper divisors and comparing it to the number:
 * 
 * @param limit 
 */
void print_abundant(const int limit)
{
    // 最小的 Abundant numbers is 12
    for (int number = 10; number <= limit; ++number)
    {
        auto sum = sum_proper_divisors(number);
        if (sum > number)
        {
            std::cout << number << ", abundance=" << sum - number << std::endl;
        }
    }
}

// -----------------------------
int main(int argc, char **argv)
{
    int limit = 0;
    std::cout << "Upper limit:";
    std::cin >> limit;

    print_abundant(limit);

    return 0;
}
