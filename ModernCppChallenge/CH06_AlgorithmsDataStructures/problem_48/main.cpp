/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief The most frequent element in a range
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

/**
 * @brief The most frequent element in a range
 * 
 * Write a function that, given a range, returns the most frequent element 
 * and the number of times it appears in the range. If more than one element appears 
 * the same maximum number of times then the function should return all the elements.
 *
 *  For instance, for the range {1,1,3,5,8,13,3,5,8,8,5}, 
 *  it should return {5, 3} and {8, 3}.
 * 
 * In order to determine and return the most frequent element in a range 
 * you should do the following:
 * 1. Count the appearances of each element in an std::map. 
 *    The key is the element and the value is its number of appearances.
 * 2. Determine the maximum element of the map using std::max_element(). 
 *    The result is a map element, that is, a pair containing the element 
 *    and its number of appearances.
 * 3. Copy all map elements that have the value (appearance count) equal to 
 *    the maximum element's value and return that as the final result.
 * 
 */

/**
 * @brief Solution:
 * An implementation of the steps described previously 
 * is shown in the following listing:
------------------------------------------------------ */
template<typename T>
std::vector<std::pair<T, size_t>> FindMostFrequent(const std::vector<T> &range)
{
    std::map<T, size_t> counts;

    for (const auto &elem : range)
    {
        counts[elem]++;
    }

    auto max_elem = std::max_element(std::cbegin(counts), std::cend(counts),
                                     [](const auto &e1, const auto &e2) { return e1.second < e2.second; });

    std::vector<std::pair<T, size_t>> result;

    std::copy_if(std::begin(counts), std::end(counts), std::back_inserter(result),
                 [max_elem](const auto &kvp) { return kvp.second == max_elem->second; });

    return result;
};

// ------------------------------
int main(int argc, char **argv)
{
    auto range = std::vector<int>{1, 1, 3, 5, 8, 13, 3, 5, 8, 8, 5};

    auto result = FindMostFrequent(range);

    for (const auto &e : result)
    {
        std::cout << e.first << " : " << e.second << std::endl;
    }

    return 0;
}
