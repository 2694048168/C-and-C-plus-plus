/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <ctime>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>


/**
 * @brief Joining strings together separated by a delimiter
 * 
 * Write a function that, given a list of strings and a delimiter, 
 * creates a new string by concatenating all the input strings separated 
 * with the specified delimiter. The delimiter must not appear after the last string, 
 * and when no input string is provided, the function must return an empty string.
 */

/**
 * @brief Solution:
--------------------------------------------------------- */
template<typename Iter>
std::string join_strings(Iter begin, Iter end, const char *const separator)
{
    std::ostringstream os;
    std::copy(begin, end - 1, std::ostream_iterator<std::string>(os, separator));

    os << *(end - 1);
    return os.str();
}

template<typename C>
std::string join_strings(const C &c, const char *const separator)
{
    if (c.size() == 0)
        return std::string{};
    return join_strings(std::begin(c), std::end(c), separator);
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

    std::vector<std::string> v1{"this", "is", "an", "example"};
    std::vector<std::string> v2{"example"};
    std::vector<std::string> v3{};

    assert(join_strings(v1, " ") == "this is an example"s);

    assert(join_strings(v2, " ") == "example"s);

    assert(join_strings(v3, " ") == ""s);

    std::array<std::string, 4> a1{
        {"this", "is", "an", "example"}
    };
    std::array<std::string, 1> a2{{"example"}};
    std::array<std::string, 0> a3{};

    assert(join_strings(a1, " ") == "this is an example"s);

    assert(join_strings(a2, " ") == "example"s);

    assert(join_strings(a3, " ") == ""s);

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
