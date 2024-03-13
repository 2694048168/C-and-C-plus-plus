/**
 * @file 20_algorithms.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代 C++ 编程学习容器之 STL 提供的标准算法库
 * @version 0.1
 * @date 2024-03-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

template<typename T>
void printContainer(const T &container)
{
    for (const auto &elem : container)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

// new beginner for C++ STL algorithm:
// https://hackingcpp.com/cpp/std/algorithms/intro.html
// ===================================
int main(int argc, const char **argv)
{
    std::vector<int> vec{7, 9, 3, 5, 3, 2, 4, 1, 8, 0};

    std::cout << "\n======== vector value ========\n";
    printContainer(vec);

    // smallest in subrange (as shown in image):
    auto iter = std::min_element(std::begin(vec) + 2, std::begin(vec) + 7);
    auto min  = *iter;
    std::cout << "The vector smallest value: " << min << '\n';

    // smallest in entire container:
    auto iter_ = std::min_element(std::begin(vec), std::end(vec));
    std::cout << "The vector smallest value: " << *iter_ << '\n';
    vec.erase(iter_); // erases smallest element

    // smallest in subrange (as shown in image):
    auto iter_max = std::max_element(std::begin(vec) + 2, std::begin(vec) + 7);
    std::cout << "The vector max value: " << *iter_max << '\n';

    // smallest in entire container:
    auto iter_max_ = std::max_element(std::begin(vec), std::end(vec));
    std::cout << "The vector max value: " << *iter_max_ << '\n';
    vec.erase(iter_max_); // erases smallest element

    std::cout << "\n======== vector erase min and max value ========\n";
    printContainer(vec);

    // C++17 supported: prefer C++17's std::reduce over std::accumulate
    unsigned int sum = std::reduce(vec.cbegin(), vec.cend(), 0);
    std::cout << "The sum of vector element: " << sum << '\n';

    // C++20 supported: invokes 'func' on each input element
    auto func = [](int &element)
    {
        ++element;
    };
    std::for_each(vec.begin(), vec.end(), func);

    sum = std::accumulate(vec.cbegin(), vec.cend(), 0);
    std::cout << "The sum of vector element: " << sum << '\n';

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\20_algorithms.cpp -std=c++23
// g++ .\20_algorithms.cpp -std=c++23
