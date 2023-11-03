/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Longest palindromic substring
 *  Write a function that, given an input string, locates and 
 * returns the longest sequence in the string that is a palindrome. 
 * If multiple palindromes of the same length exist, 
 * the first one should be returned.
 * 
 */

/**
 * @brief Solution:
---------------------------------------------- */

std::string longest_palindrome(std::string_view str)
{
    const size_t len          = str.size();
    size_t       longestBegin = 0;
    size_t       maxLen       = 1;

    std::vector<bool> table(len * len, false);

    for (size_t i = 0; i < len; i++)
    {
        table[i * len + i] = true;
    }

    for (size_t i = 0; i < len - 1; i++)
    {
        if (str[i] == str[i + 1])
        {
            table[i * len + i + 1] = true;
            if (maxLen < 2)
            {
                longestBegin = i;
                maxLen       = 2;
            }
        }
    }

    for (size_t k = 3; k <= len; k++)
    {
        for (size_t i = 0; i < len - k + 1; i++)
        {
            size_t j = i + k - 1;
            if (str[i] == str[j] && table[(i + 1) * len + j - 1])
            {
                table[i * len + j] = true;
                if (maxLen < k)
                {
                    longestBegin = i;
                    maxLen       = k;
                }
            }
        }
    }

    return std::string(str.substr(longestBegin, maxLen));
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime()
{
    time_t    now = time(0);
    struct tm tstruct;
    char      buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

// --------------------------------
int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    using namespace std::string_literals;

    assert(longest_palindrome("sahararahnide") == "hararah");
    assert(longest_palindrome("level") == "level");
    assert(longest_palindrome("s") == "s");
    assert(longest_palindrome("aabbcc") == "aa");
    assert(longest_palindrome("abab") == "aba");

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    auto duration_us = std::chrono::duration<double, std::micro>(end - start).count();
    auto duration_ns = std::chrono::duration<double, std::nano>(end - start).count();
    auto duration_s  = std::chrono::duration<double>(end - start).count();

    std::cout << "===============================\n";
    std::cout << "[Time Consumption] " << duration_s << " s\n";
    std::cout << "[Time Consumption] " << duration_ms << " ms\n";
    std::cout << "[Time Consumption] " << duration_us << " us\n";
    std::cout << "[Time Consumption] " << duration_ns << " ns\n";
    std::cout << "===============================\n";

    return 0;
}
