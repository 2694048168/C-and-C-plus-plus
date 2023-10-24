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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

/* Prime factors of a number
Write a program that prints the prime factors of a number entered by the user.
--------------------------------------- */

/* Solution
The prime factors of a positive integer are the prime numbers that divide 
that integer exactly. For instance, the prime factors of 8 are 2 x 2 x 2, 
and the prime factors of 42 are 2 x 3 x 7. 
To determine the prime factors you should use the following algorithm: 
------------------------------------------------------------------------
Step1. While n is divisible by 2, 2 is a prime factor and must be added to the list,
    while n becomes the result of n/2. 
    After completing this step, n is an odd number.

Step2. Iterate from 3 to the square root of n. While the current number, 
    let’s call it i, divides n, i is a prime factor and must be added to the list,
    while n becomes the result of n/i. 
    When i no longer divides n, increment i by 2 (to get the next odd number).    

Step3. When n is a prime number greater than 2, the steps above will not result 
    in n becoming 1. Therefore, if at the end of step 2 n is still greater than 2,
    then n is a prime factor.    
----------------------------------------------------------- */
std::vector<unsigned long long> prime_factors(unsigned long long n)
{
    std::vector<unsigned long long> factors;
    // print the number of 2s that divide n
    while (n % 2 == 0)
    {
        factors.push_back(2);
        n = n / 2;
    }

    for (unsigned long long i = 3; i <= std::sqrt(n); i += 2)
    {
        // while i divides n, print i and divide n
        while (n % i == 0)
        {
            factors.push_back(i);
            n = n / i;
        }
    }

    // n is a prime number greater than 2
    if (n > 2)
        factors.push_back(n);

    return factors;
}

// -----------------------------
int main(int argc, char **argv)
{
    unsigned long long number = 0;
    std::cout << "Please enter the number: ";
    std::cin >> number;

    auto factors = prime_factors(number);

    std::copy(std::begin(factors), std::end(factors), std::ostream_iterator<unsigned long long>(std::cout, " "));

    return 0;
}
