/**
 * @file non_manipulating_algorithms_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

template<typename T>
bool print_container(const T &container, const char *msg)
{
    if (container.empty())
    {
        std::cout << "the container is empty, please check.\n" << std::endl;
        return false;
    }

    std::printf("%s: ", msg);
    for (const auto elem : container)
    {
        std::cout << elem << " ";
    }
    std::printf("\n");

    return true;
}

/**
 * @brief the Non-Manipulating Algorithms in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------------------------------------
    std::vector<int> vec1{10, 20, 5, 23, 42, 15};
    print_container(vec1, "the elements of vector: ");

    std::sort(vec1.begin(), vec1.end());
    print_container(vec1, "after std::sort(default): ");

    std::sort(vec1.begin(), vec1.end(), std::greater<int>());
    print_container(vec1, "after std::sort(greater): ");

    std::reverse(vec1.begin(), vec1.end());
    print_container(vec1, "after std::reverse: ");

    // Starting the summation from 0
    std::cout << "\nThe summation of vector elements is: ";
    std::cout << std::accumulate(vec1.begin(), vec1.end(), 0) << std::endl;

    // -------------------------------------
    std::cout << "--------------------------" << std::endl;
    std::vector<int> vec2{10, 20, 5, 20, 23, 42, 15};
    std::cout << "Occurrences of 20 in vector: ";
    std::cout << std::count(vec2.begin(), vec2.end(), 20) << std::endl;

    if (std::find(vec2.begin(), vec2.end(), 42) != vec2.end())
    {
        std::cout << "42 found in the vector." << std::endl;
    }
    else
    {
        std::cout << "42 NOT found in the vector." << std::endl;
    }

    std::cout << "--------------------------" << std::endl;
    std::vector<int> arr{5, 10, 15, 20, 20, 23, 42, 45};

    if (std::binary_search(arr.begin(), arr.end(), 44))
    {
        std::cout << "44 found in the vector." << std::endl;
    }
    else
    {
        std::cout << "44 NOT found in the vector." << std::endl;
    }

    // Returns the first occurrence of 20
    auto iter_lower = std::lower_bound(arr.begin(), arr.end(), 20);

    // Returns the last occurrence of 20
    auto iter_uppper = std::upper_bound(arr.begin(), arr.end(), 20);

    std::cout << "the lower bound is at position: ";
    // std::distance(arr.begin(), iter_lower)
    std::cout << (iter_lower - arr.begin()) << std::endl;

    std::cout << "the upper bound is at position: ";
    std::cout << std::distance(arr.begin(), iter_uppper) << std::endl;

    return 0;
}