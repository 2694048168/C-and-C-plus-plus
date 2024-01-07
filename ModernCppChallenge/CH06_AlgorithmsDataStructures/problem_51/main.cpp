/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Transforming a list of phone numbers
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Transforming a list of phone numbers
 * 
 * Write a function that, given a list of phone numbers, 
 * transforms them so they all start with a specified phone country code, 
 * preceded by the + sign. Any whitespaces from a phone number should also be removed.
 * 
 * This problem is somewhat similar in some aspects to the previous one. 
 * However, instead of selecting phone numbers that start with a specified country code,
 * we must transform each number so that they all start with that country code preceded
 *  by a +. There are several cases that must be considered:
 * 1. The phone number starts with a 0. That indicates a number without a country code.
 *   To modify the number to include the country code we must replace the 0 with
 *    the actual country code, preceded by +.
 * 2. The phone number starts with the country code. In this case, 
 *    we just prepend + sign to the beginning.
 * 3. The phone number starts with + followed by the country code. In this case, 
 *    the number is already in the expected format.
 * 4. None of these cases applies, therefore the result is obtained by concatenating 
 *    the country code preceded by + and the phone number. 
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */

bool starts_with(std::string_view str, std::string_view prefix)
{
    return str.find(prefix) == 0;
}

void normalize_phone_numbers(std::vector<std::string> &numbers, const std::string &countryCode)
{
    std::transform(std::cbegin(numbers), std::cend(numbers), std::begin(numbers),
                   [countryCode](const std::string &number)
                   {
                       std::string result;
                       if (number.size() > 0)
                       {
                           if (number[0] == '0')
                               result = "+" + countryCode + number.substr(1);
                           else if (starts_with(number, countryCode))
                               result = "+" + number;
                           else if (starts_with(number, "+" + countryCode))
                               result = number;
                           else
                               result = "+" + countryCode + number;
                       }

                       result.erase(std::remove_if(std::begin(result), std::end(result),
                                                   [](const char ch) { return isspace(ch); }),
                                    std::end(result));

                       return result;
                   });
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<std::string> numbers{"07555 123456", "07555123456", "+44 7555 123456", "44 7555 123456", "7555 123456"};

    normalize_phone_numbers(numbers, "44");

    for (const auto &number : numbers)
    {
        std::cout << number << std::endl;
    }

    return 0;
}
