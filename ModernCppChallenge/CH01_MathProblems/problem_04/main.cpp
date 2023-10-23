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

/*  Largest prime smaller than given number 小于给定数的最大素数
Write a program that computes and prints the largest prime number 
that is smaller than a number provided by the user, 
which must be a positive integer.
------------------------------------------- */

/* Solution 素数 (Prime number)
质数，又称素数，指在大于1的自然数中，
除了1和该数自身外，无法被其他自然数整除的数。
------------------------------------ */
bool is_prime(const int num)
{
    if (num <= 3)
    {
        // 2 and 3
        return num > 1;
    }
    else if (num % 2 == 0 || num % 3 == 0)
    {
        // 被 2 或 3 整除的数, 肯定不是 prime number
        return false;
    }
    else
    {
        // 然后从 5 开始
        /* 质数特点, 总是等于 6x-1 或者 6x+1, 其中 x 是大于等于1的自然数
        step 1. 6x 肯定不是质数，因为它能被 6 整除；
        step 2. 6x+2 肯定也不是质数，因为它还能被2整除；
        step 3. 6x+3 肯定能被 3 整除；
        step 4. 6x+4 肯定能被 2 整除。
        step 5. 6x+1 和 6x+5 (即等同于6x-1) 可能是质数了
        所以循环的步长可以设为 6，然后每次只判断 6 两侧的数即可。
        ------------------------------------------------- */
        for (int x = 5; x * x <= num; x += 6)
        {
            if (num % x == 0 || num % (x + 2) == 0)
            {
                return false;
            }
        }

        return true;
    }
}

// -----------------------------
int main(int argc, char **argv)
{
    int limit = 0;
    std::cout << "Upper limit: ";
    std::cin >> limit;

    for (int i = limit; i > 1; --i)
    {
        if (is_prime(i))
        {
            std::cout << "Largest prime: " << i << std::endl;

            return 0;
        }
    }
}
