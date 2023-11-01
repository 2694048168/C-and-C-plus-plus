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

#include <cassert>
#include <cctype>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>


/**
 * @brief Capitalizing an article title
 * Write a function that transforms an input text into a capitalized version, 
 * where every word starts with an uppercase letter and 
 * has all the other letters in lowercase. 
 * For instance, the text "the c++ challenger" should be transformed to "The C++ Challenger".
 * 
 */

/**
 * @brief Solution:

The function template capitalize(), implemented as follows, 
works with strings of any type of characters. 
It does not modify the input string but creates a new string. 
To do so, it uses an std::stringstream. 
It iterates through all the characters in the input string and sets a flag indicating 
a new word to true every time a space or punctuation is encountered.
Input characters are transformed to uppercase when they represent 
the first character in a word and to lowercase otherwise:
--------------------------------------------------------- */
template<class Elem>
using tstring = std::basic_string<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<class Elem>
using tstringstream = std::basic_stringstream<Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template<class Elem>
tstring<Elem> capitalize(const tstring<Elem> &text)
{
    tstringstream<Elem> result;

    bool newWord = true;
    for (const auto ch : text)
    {
        newWord = newWord || std::ispunct(ch) || std::isspace(ch);
        if (std::isalpha(ch))
        {
            if (newWord)
            {
                result << static_cast<Elem>(std::toupper(ch));
                newWord = false;
            }
            else
                result << static_cast<Elem>(std::tolower(ch));
        }
        else
            result << ch;
    }

    return result.str();
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime()
{
    time_t     now = time(0);
    struct tm tstruct;
    char       buf[80];
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

    std::string text = "THIS IS an ExamplE, should wORk!";

    std::string expected = "This Is An Example, Should Work!";

    assert(expected == capitalize(text));

    assert(L"The C++ Challenger"s == capitalize(L"the c++ challenger"s));

    assert(L"This Is An Example, Should Work!"s == capitalize(L"THIS IS an ExamplE, should wORk!"s));

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
