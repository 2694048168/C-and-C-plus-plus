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
#include <vector>


/**
 * @brief License plate validation
 *  Considering license plates with the format LLL-LL DDD or LLL-LL DDDD 
 * (where L is an uppercase letter from A to Z and D is a digit), write:
 * One function that validates that a license plate number is of the correct format;
 * One function that, given an input text, extracts 
 *  and returns all the license plate numbers found in the text
 */

/**
 * @brief Solution:

 The simplest way to solve this problem is by using regular expressions. 
The regular expression that meets the described format is "[A-Z]{3}-[A-Z]{2} \d{3,4}".
The regular expression would therefore change to "([A-Z]{3}-[A-Z]{2} \d{3,4})*".
---------------------------------------------- */
bool validate_license_plate_format(std::string_view str)
{
    std::regex rx(R"([A-Z]{3}-[A-Z]{2} \d{3,4})");

    return std::regex_match(str.data(), rx);
}

std::vector<std::string> extract_license_plate_numbers(const std::string &str)
{
    std::vector<std::string> results;

    std::regex  rx(R"(([A-Z]{3}-[A-Z]{2} \d{3,4})*)");
    std::smatch match;

    for (auto i = std::sregex_iterator(std::cbegin(str), std::cend(str), rx); i != std::sregex_iterator(); ++i)
    {
        if ((*i)[1].matched)
            results.push_back(i->str());
    }

    return results;
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
    assert(validate_license_plate_format("ABC-DE 123"));
    assert(validate_license_plate_format("ABC-DE 1234"));
    assert(!validate_license_plate_format("ABC-DE 12345"));
    assert(!validate_license_plate_format("abc-de 1234"));

    std::vector<std::string> expected{"AAA-AA 123", "ABC-DE 1234", "XYZ-WW 0001"};
    std::string              text("AAA-AA 123qwe-ty 1234  ABC-DE 123456..XYZ-WW 0001");
    assert(expected == extract_license_plate_numbers(text));

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
