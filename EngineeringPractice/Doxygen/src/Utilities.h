/**
 * @file Utilities.h
 * @brief Provides utility functions used throughout the application.
 */

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Splits a string into substrings based on the specified delimiter.
 * @param str The string to split.
 * @param delimiter The character used to split the string.
 * @return A vector of substrings.
 */
std::vector<std::string> split(const std::string &str, char delimiter);

/**
 * @class StringUtils
 * @brief Provides a collection of static methods for string manipulation.
 *
 * StringUtils class offers various static methods that can be used to perform
 * common string operations, such as trimming, converting to uppercase, and checking
 * if a string is numeric.
 */
class StringUtils
{
public:
    /**
     * @brief Checks if the given string consists only of numeric characters.
     * @param s The string to check.
     * @return True if the string is numeric, false otherwise.
     */
    static bool isNumeric(const std::string &s)
    {
        return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    }
};
