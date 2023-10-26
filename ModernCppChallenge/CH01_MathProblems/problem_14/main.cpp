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

#include <algorithm>
#include <cassert>
#include <ios>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string_view>

/* Validating ISBNs
Write a program that validates that 10-digit values entered by the user, 
as a string, represent valid ISBN-10 numbers.
------------------------------------------------ */

/* Solution
The International Standard Book Number (ISBN) is a unique numeric identifier for books.
Currently, a 13-digit format is used. However, for this problem, you are to validate the
former format that used 10 digits. 

The last of the 10 digits is a checksum. 
This digit is chosen so that the sum of all the ten digits, 
each multiplied by its (integer) weight,
descending from 10 to 1, is a multiple of 11.
--------------------------------------------------- */
bool validate_isbn_10(std::string_view isbn)
{
    bool valid = false;
    if (isbn.size() == 10 && std::count_if(std::begin(isbn), std::end(isbn), isdigit) == 10)
    {
        auto w = 10;

        auto sum = std::accumulate(std::begin(isbn), std::end(isbn), 0,
                                   [&w](const int total, const char c) { return total + w-- * (c - '0'); });

        valid = !(sum % 11);
    }

    return valid;
}

// -----------------------------
int main(int argc, char **argv)
{
    assert(validate_isbn_10("0306406152"));
    assert(!validate_isbn_10("0306406151"));

    // static_assert(validate_isbn_10("0306406152"), "valid");
    // static_assert(!validate_isbn_10("0306406151"), "not valid");

    std::string isbn;
    std::cout << "isbn:";
    std::cin >> isbn;

    std::cout << "valid: " << validate_isbn_10(isbn) << std::endl;
    std::cout << "valid: " << std::boolalpha << validate_isbn_10(isbn) << std::endl;

    return 0;
}
