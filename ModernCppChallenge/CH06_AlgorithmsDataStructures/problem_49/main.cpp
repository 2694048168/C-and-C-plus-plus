/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Text histogram
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cctype>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <string_view>
#include <algorithm>

/**
 * @brief Text histogram
 * 
 * Write a program that, given a text, determines and prints a histogram with the frequency
 * of each letter of the alphabet. The frequency is the percentage of the number of
 * appearances of each letter from the total count of letters. The program should
 * count only the appearances of letters and ignore digits, signs, 
 * and other possible characters. The frequency must be determined based on 
 * the count of letters and not the text size.
 * 
 * A histogram is a representation of the distribution of numerical data. 
 * Widely known histograms are the color and image histograms that are used 
 * in photography and image processing. A text histogram, as described here, 
 * is a representation of the frequency of letters in a given text. 
 * This problem is partially similar to the previous one, except that the range elements 
 * are characters now and we must determine the frequency of them all. 
 * 
 * To solve this problem you should:
 * 1. Count the appearances of each letter using a map. 
 *    The key is the letter and the value is its appearance count.
 * 2. When counting, ignore all characters that are not letters. Uppercase and lowercase\
 *    characters must be treated as identical, as they represent the same letter.
 * 3. Use std::accumulate() to count the total number of appearances of all 
 *    the letters in the given text.
 * 4. Use std::for_each() or a range-based for loop to go through all the elements
 *    of the map and transform the appearance count into a frequency.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
std::map<char, double> analyzeText(std::string_view text)
{
    std::map<char, double> frequencies;

    for (char ch = 'a'; ch <= 'z'; ++ch)
    {
        frequencies[ch] = 0;
    }

    for (auto ch : text)
    {
        if (std::isalpha(ch))
            frequencies[std::tolower(ch)]++;
    }

    auto total = std::accumulate(std::cbegin(frequencies), std::cend(frequencies), 0ull,
                                 [](const auto sum, const auto &kvp)
                                 { return sum + static_cast<unsigned long long>(kvp.second); });

    std::for_each(std::begin(frequencies), std::end(frequencies),
                  [total](auto &kvp) { kvp.second = (100.0 * kvp.second) / total; });
    return frequencies;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto result = analyzeText(
        R"(Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.)");

    for (const auto &kvp : result)
    {
        std::cout << kvp.first << " : " << std::fixed << std::setw(5) << std::setfill(' ') << std::setprecision(2)
                  << kvp.second << std::endl;
    }

    return 0;
}
