/**
 * @file numeric_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <numeric>
#include <vector>

int myfun(int x, int y)
{
    return x * y;
}

/**
 * @brief The numeric header is part of the numeric library in C++ STL.
 *   This library consists of basic mathematical functions and types,
 *   as well as optimized numeric arrays and support for random number generation.
 * 
 * iota | accumulate | reduce | inner_product | partial_sum
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // --------------------------------
    std::vector<int> vect{5, 10, 15};

    auto sum = std::accumulate(vect.begin(), vect.end(), 0);
    std::cout << "the sum of vector: " << sum << std::endl;

    auto product = std::accumulate(vect.begin(), vect.end(), 1, myfun);
    std::cout << "the product of vector: " << product << std::endl;

    std::cout << "the minus of vector: ";
    std::cout << std::accumulate(vect.begin(), vect.end(), 0, std::minus<int>());

    // --------------------------------
    // TODO std::partial_sum
    // TODO std::iota
    // TODO std::reduce
    // TODO std::inner_product

    return 0;
}