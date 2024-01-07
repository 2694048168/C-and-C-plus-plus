/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Pairwise algorithm
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <vector>

/**
 * @brief Pairwise algorithm
 * 
 * Write a general-purpose function that, given a range, 
 * returns a new range with pairs of consecutive elements from the input range. 
 * Should the input range have an odd number of elements, the last one must be ignored.
 * 
 * The pairwise function proposed for this problem must pair adjacent elements of
 * an input range and produce std::pair elements that are added to an output range. 
 * The following code listing provides two implementations:
 * 1. A general function template that takes iterators as arguments: a begin and end
 *   iterator define the input range, and an output iterator defines the position 
 *   in the output range where the results are to be inserted
 * 2. An overload that takes an std::vector<T> as the input argument and returns an
 *   std::vector<std::pair<T, T>> as the result; 
 * this one simply calls the first overload:
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename Input, typename Output>
void pairwise(Input begin, Input end, Output result)
{
    auto it = begin;
    while (it != end)
    {
        auto v1 = *it++;
        if (it == end)
            break;
        auto v2  = *it++;
        result++ = std::make_pair(v1, v2);
    }
}

template<typename T>
std::vector<std::pair<T, T>> pairwise(const std::vector<T> &range)
{
    std::vector<std::pair<T, T>> result;
    pairwise(std::begin(range), std::end(range), std::back_inserter(result));

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<int> v{1, 1, 3, 5, 8, 13, 21};

    auto result = pairwise(v);

    for (const auto &p : result)
    {
        std::cout << '{' << p.first << ',' << p.second << '}' << std::endl;
    }

    return 0;
}
