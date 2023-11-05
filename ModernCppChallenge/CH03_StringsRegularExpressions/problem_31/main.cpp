/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cassert>
#include <ctime>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>

/**
 * @brief Transforming dates in strings
 *  Write a function that, given a text containing dates in the format
 * dd.mm.yyyy or dd-mmyyyy, 
 * transforms the text so that it contains dates in the format yyyy-mm-dd.
 */

/**
 * @brief Solution:

 Text transformation can be performed with regular expressions using
std::regex_replace(). A regular expression that can match dates with the specified
formats is (\d{1,2})(\.|-|/)(\d{1,2})(\.|-|/)(\d{4}). 
------------------------------------------------------ */
std::string transform_date(std::string_view text)
{
    auto rx = std::regex{R"((\d{1,2})(\.|-|/)(\d{1,2})(\.|-|/)(\d{4}))"};
    return std::regex_replace(text.data(), rx, R"($5-$3-$1)");
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
    using namespace std::string_literals;

    assert(transform_date("today is 01.12.2017!"s) == "today is 2017-12-01!"s);

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
