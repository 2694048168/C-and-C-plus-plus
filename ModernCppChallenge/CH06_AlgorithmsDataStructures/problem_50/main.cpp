/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Filtering a list of phone numbers
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>

/**
 * @brief Filtering a list of phone numbers
 * 
 * Write a function that, given a list of phone numbers, returns only the numbers 
 * that are from a specified country. The country is indicated by its phone country code,
 * such as 44 for Great Britain. Phone numbers may start with the country code, 
 * a + followed by the country code, or have no country code. 
 * The ones from this last category must be ignored.
 * 
 * The solution to this problem is relatively simple:
 * you have to iterate through all the phone numbers and copy to a separate container
 * (such as an std::vector) the phone numbers that start with the country code.
 * If the specified country code is, for instance, 44, then you must check for 
 * both 44 and +44. Filtering the input range in this manner is possible using
 * the std::copy_if() function.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
bool start_with(std::string_view str, std::string_view prefix)
{
    return str.find(prefix) == 0;
}

template<typename InputIt>
std::vector<std::string> filter_numbers(InputIt begin, InputIt end, const std::string &countryCode)
{
    std::vector<std::string> result;

    std::copy_if(begin, end, std::back_inserter(result),
                 [countryCode](const auto &number)
                 { return start_with(number, countryCode) || start_with(number, "+" + countryCode); });

    return result;
}

std::vector<std::string> filter_numbers(const std::vector<std::string> &numbers, const std::string &countryCode)
{
    return filter_numbers(std::cbegin(numbers), std::cend(numbers), countryCode);
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<std::string> numbers{"+40744909080", "44 7520 112233", "+44 7555 123456", "40 7200 123456",
                                     "7555 123456"};

    auto result = filter_numbers(numbers, "44");

    for (const auto &number : result)
    {
        std::cout << number << std::endl;
    }

    return 0;
}
