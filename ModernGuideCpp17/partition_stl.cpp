/**
 * @file partition_stl.cpp
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
 * @brief STL algorithms library which allows us easy partition algorithms
 *        using certain inbuilt functions.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -----------------------------------
    std::vector<int> vect = {2, 1, 5, 6, 8, 7};

    std::is_partitioned(vect.begin(), vect.end(), [](int x) { return x % 2 == 0; })
        ? std::cout << "this vector is partitioned"
        : std::cout << "this vector is NOT partitioned";

    std::cout << "\n" << std::endl;

    // -----------------------------------
    std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    print_container(v, "the elements of vector: ");

    auto it = std::partition(v.begin(), v.end(), [](int i) { return i % 2 == 0; });

    print_container(v, "after partition: ");

    return 0;
}