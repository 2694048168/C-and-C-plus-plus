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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Splitting a string into tokens with a list of possible delimiters
 *  Write a function that, given a string and a list of possible delimiter characters,
 * splits the string into tokens separated by any of the delimiters 
 * and returns them in an std::vector.
 * 
 */

/**
 * @brief Solution:

Two different versions of a splitting function are listed as follows:
The first one uses a single character as the delimiter. 
 To split the input string it uses a string stream initialized 
 with the content of the input string, using std::getline() to 
 read chunks from it until the next delimiter or an end-of-line character is encountered.

The second one uses a list of possible character delimiters, specified in an std::string.
 It uses std:string::find_first_of() to locate the first position 
 of any of the delimiter characters, starting from a given position. 
 It does so in a loop until the entire input string is being processed.
 
The extracted substrings are added to the result vector:
---------------------------------------------- */
template<class Elem>
using tstring = std::basic_string<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<class Elem>
using tstringstream = std::basic_stringstream<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<typename Elem>
inline std::vector<tstring<Elem>> split(tstring<Elem> text, const Elem delimiter)
{
    auto sstr   = tstringstream<Elem>{text};
    auto tokens = std::vector<tstring<Elem>>{};
    auto token  = tstring<Elem>{};

    while (std::getline(sstr, token, delimiter))
    {
        if (!token.empty())
            tokens.push_back(token);
    }

    return tokens;
}

template<typename Elem>
inline std::vector<tstring<Elem>> split(tstring<Elem> text, const tstring<Elem> &delimiters)
{
    auto tokens = std::vector<tstring<Elem>>{};

    size_t position;
    size_t prev_postion = 0;
    while ((position = text.find_first_of(delimiters, prev_postion)) != std::string::npos)
    {
        if (position > prev_postion)
            tokens.push_back(text.substr(prev_postion, position - prev_postion));

        prev_postion = position + 1;
    }

    if (prev_postion < text.length())
        tokens.push_back(text.substr(prev_postion, std::string::npos));

    return tokens;
}

template<typename T>
void print_container(std::vector<T> &container)
{
    // for (const auto elem : container)
    std::copy(std::begin(container), std::end(container), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}

// --------------------------------
int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    using namespace std::string_literals;
    std::string              test_str = "this,is.a sample!!";
    std::vector<std::string> expected{"this", "is", "a", "sample"};

    //  output: {"this", "is", "a", "sample"}
    std::vector<std::string> expected_str = split(test_str, ",.! "s);
    print_container(expected_str);
    assert(expected_str == expected);

    assert(expected == split("this is a sample"s, ' '));
    assert(expected == split("this,is a.sample!!"s, ",.! "s));

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
