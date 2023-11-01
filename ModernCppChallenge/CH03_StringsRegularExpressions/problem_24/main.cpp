/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cassert>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

/**
 * @brief String to binary conversion
 * Write a function that, given a string containing hexadecimal digits 
 * as the input argument, returns a vector of 8-bit integers that 
 * represent the numerical deserialization of the string content.
 * 
 */

/**
 * @brief Solution:
The operation requested here is the opposite of the one implemented 
in the previous problem. 
This time, however, we could write a function and not a function template.
The input is an std::string_view, 
 which is a lightweight wrapper for a sequence of characters. 
The output is a vector of 8-bit unsigned integers.

The following hexstr_to_bytes function transforms every two text characters into
 an unsigned char value ("A0" becomes 0xA0), 
 puts them into an std::vector, and returns the vector:
--------------------------------------------------------- */
unsigned char hexchar_to_int(const char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;

    throw std::invalid_argument("Invalid hexadecimal character");
}

std::vector<unsigned char> hexstr_to_bytes(std::string_view str)
{
    std::vector<unsigned char> result;

    for (size_t i = 0; i < str.size(); i += 2)
    {
        result.push_back((hexchar_to_int(str[i]) << 4) | hexchar_to_int(str[i + 1]));
    }

    return result;
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime()
{
    time_t     now = time(0);
    struct tm tstruct;
    // struct tm *tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // localtime_s(tstruct, &now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    // strftime(buf, sizeof(buf), "%Y-%m-%d %X", tstruct);

    return buf;
}

// --------------------------------
int main(int argc, char **argv)
{
    // ! why the localtime NOT show?
    std::cout << "[" << currentDateTime() << "] ======== Begin test ========\n";

    std::vector<unsigned char> expected{0xBA, 0xAD, 0xF0, 0x0D, 0x42};

    assert(hexstr_to_bytes("BAADF00D42") == expected);
    assert(hexstr_to_bytes("BaaDf00d42") == expected);

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
