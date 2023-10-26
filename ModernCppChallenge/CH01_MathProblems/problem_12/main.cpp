/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

/* Largest Collatz sequence
Write a program that determines and prints which number up to 1 million produces the
longest Collatz sequence and what its length is.
------------------------------------------------- */

/* Solution
The Collatz conjecture, also known as the Ulam conjecture, Kakutani's problem, 
the Thwaites conjecture, Hasse's algorithm, or the Syracuse problem, 
is an unproven conjecture that states that a sequence defined as explained 
in the following always reaches 1. The series is defined as follows: 
start with any positive integer n and obtain each new term from the previous one: 
if the previous term is even, the next term is half the previous term, 
or else it is 3 times the previous term plus 1.

The problem you are to solve is to generate the Collatz sequence for all positive integers up
to one million, determine which of them is the longest, and print its length and the starting
number that produced it. Although we could apply brute force to generate the sequence for
each number and count the number of terms until reaching 1, a faster solution would be to
save the length of all the sequences that have already been generated. When the current
term of a sequence that started from a value n becomes smaller than n, then it is a number
whose sequence has already been determined, so we could simply fetch its cached length
and add it to the current length to determine the length of the sequence started from n. This
approach, however, introduces a limit to the Collatz sequences that could be computed,
because at some point the cache will exceed the amount of memory the system can allocate:
----------------------------------------------- */
std::pair<unsigned long long, long> longest_collatz_uncached(const unsigned long long limit)
{
    long               length = 0;
    unsigned long long number = 0;

    for (unsigned long long i = 2; i <= limit; i++)
    {
        auto n     = i;
        long steps = 0;
        while (n != 1)
        {
            if ((n % 2) == 0)
                n = n / 2;
            else
                n = n * 3 + 1;
            steps++;
        }

        if (steps > length)
        {
            length = steps;
            number = i;
        }
    }

    return std::make_pair(number, length);
}

std::pair<unsigned long long, long> longest_collatz(const unsigned long long limit)
{
    long               length = 0;
    unsigned long long number = 0;

    std::vector<int> cache(limit + 1, 0);

    for (unsigned long long i = 2; i <= limit; i++)
    {
        auto n     = i;
        long steps = 0;
        while (n != 1 && n >= i)
        {
            if ((n % 2) == 0)
                n = n / 2;
            else
                n = n * 3 + 1;
            steps++;
        }
        cache[i] = steps + cache[n];

        if (cache[i] > length)
        {
            length = cache[i];
            number = i;
        }
    }

    return std::make_pair(number, length);
}

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

void test_demo()
{
    struct test_data
    {
        unsigned long long limit;
        unsigned long long start;
        long               steps;
    };

    std::vector<test_data> data{
        {       10ULL,        9ULL,  19},
        {      100ULL,       97ULL, 118},
        {     1000ULL,      871ULL, 178},
        {    10000ULL,     6171ULL, 263},
        {   100000ULL,    77031ULL, 350},
        {  1000000ULL,   837799ULL, 524},
        { 10000000ULL,  8400511ULL, 685},
        {100000000ULL, 63728127ULL, 949}
    };

    for (const auto &d : data)
    {
        auto result = longest_collatz(d.limit);

        if (result.first != d.start || result.second != d.steps)
            std::cout << "error on limit " << d.limit << std::endl;
        else
            std::cout << "less than      : " << d.limit << std::endl
                      << "starting number: " << result.first << std::endl
                      << "sequence length: " << result.second << std::endl;
    }
}

// -----------------------------
int main(int argc, char **argv)
{
    std::cout << "=====================================\n";
    // 测试耗时
    auto timer1 = perf_timer<>::duration(test_demo);

    std::cout << "=====================================\n";
    std::cout << std::chrono::duration<double, std::milli>(timer1).count() << " ms\n";

    return 0;
}
