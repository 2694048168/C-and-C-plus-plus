/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Zip algorithm
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <vector>

/**
 * @brief Zip algorithm
 * 
 * Write a function that, given two ranges, returns a new range with pairs of
 * elements from the two ranges. Should the two ranges have different sizes, 
 * the result must contain as many elements as the smallest of the input ranges.
 * 
 * This problem is relatively similar to the previous one, although there are 
 * two input ranges instead of just one. The result is again a range of std::pair.
 * However, the two input ranges may hold elements of different types. 
 * Again, the implementation shown here contains two overloads:
 * 1. A general-purpose function with iterators as arguments. A begin and end iterator
 *   for each input range define its bounds, and an output iterator defines the
 *   position in the output range where the result must be written.
 * 2. A function that takes two std::vector arguments, one that holds elements of
 *   type T and one that holds elements of type U and returns
 *   an std::vector<std::pair<T, U>>. This overload simply calls the previous one:
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename Input1, typename Input2, typename Output>
void zip(Input1 begin1, Input1 end1, Input2 begin2, Input2 end2, Output result)
{
    auto it1 = begin1;
    auto it2 = begin2;
    while (it1 != end1 && it2 != end2)
    {
        result++ = std::make_pair(*it1++, *it2++);
    }
}

template<typename T, typename U>
std::vector<std::pair<T, U>> zip(const std::vector<T> &range1, const std::vector<U> &range2)
{
    std::vector<std::pair<T, U>> result;

    zip(std::begin(range1), std::end(range1), std::begin(range2), std::end(range2), std::back_inserter(result));

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<int> v1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v2{1, 1, 3, 5, 8, 13, 21};

    auto result = zip(v1, v2);

    for (const auto &p : result)
    {
        std::cout << '{' << p.first << ',' << p.second << '}' << std::endl;
    }

    return 0;
}
