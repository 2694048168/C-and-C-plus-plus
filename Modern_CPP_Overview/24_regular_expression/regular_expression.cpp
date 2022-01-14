/**
 * @file regular_expression.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief regular expression; std::regex
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <string>
#include <regex>

int main(int argc, char **argv)
{
    std::string file_names[] = {"foo.txt", "bar.txt", "test", "a0.txt", "AAA.txt"};
    /* In C++, `\` will be used as an escape character in the string. 
    In order for `\.` to be passed as a regular expression, 
    it is necessary to perform second escaping of `\`, thus we have `\\.` */
    std::regex txt_regex("[a-z]+\\.txt");
    for (const auto &fname : file_names)
    {
        std::cout << fname << ": " << std::regex_match(fname, txt_regex) << std::endl;
    }

    std::regex base_regex("([a-z]+)\\.txt");
    std::smatch base_match;
    for (const auto &fname : file_names)
    {
        if (std::regex_match(fname, base_match, base_regex))
        {
            /* the first element of std::smatch matches the entire string
            the second element of std::smatch matches the first expression with brackets */
            if (base_match.size() == 2)
            {
                std::string base = base_match[1].str();
                std::cout << "sub-match[0]: " << base_match[0].str() << std::endl;
                std::cout << fname << " sub-match[1]: " << base << std::endl;
            }
        }
    }

    return 0;
}
