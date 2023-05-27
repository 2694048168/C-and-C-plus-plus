/**
 * @file manipulating_algorithms_stl.cpp
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
 * @brief some manipulating algorithms in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ---------------------------------------
    std::vector<int> vect{5, 10, 15, 20, 20, 23, 42, 45};
    print_container(vect, "the elements of vector: ");

    vect.erase(std::find(vect.begin(), vect.end(), 10));
    print_container(vect, "after erasing element: ");

    vect.erase(std::unique(vect.begin(), vect.end()), vect.end());
    print_container(vect, "after removing duplicates: ");

    // ---------------------------------------
    std::cout << "----------------------------" << std::endl;
    std::vector<char> vec_ch{'a', 'b', 'c'};
    print_container(vec_ch, "the element of vector: ");

    // modifies vector to its next permutation order
    do
    {
        print_container(vec_ch, "after performing next permutation: ");
    }
    while (std::next_permutation(vec_ch.begin(), vec_ch.end()));

    std::prev_permutation(vec_ch.begin(), vec_ch.end());
    print_container(vec_ch, "after performing prev permutation: ");

    // ---------------------------------------
    // std::distance() is very useful while finding the index!
    std::cout << "----------------------------" << std::endl;
    std::vector<int> vec3{5, 10, 15, 20, 20, 23, 42, 45};
    print_container(vec3, "the elements of vector: ");

    auto index_max = std::distance(vec3.begin(), std::max_element(vec3.begin(), vec3.end()));

    std::cout << "Distance between first to max element: " << index_max << std::endl;

    return 0;
}