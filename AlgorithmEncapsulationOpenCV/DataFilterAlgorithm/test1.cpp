/**
 * @file test1.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-19
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ test1.cpp -std=c++20
 * clang++ test1.cpp -std=c++20
 *  
 */

#include "LimitedAmplitudeFilter.hpp"

#include <iostream>
#include <vector>

// -------------------------------------
int main(int argc, const char *argv[])
{
    std::vector<double> dataDouble{22.1, 22.4, 22.3, 22.0, 222, 32.9, 22.5};

    LimitedAmplitudeFilter<double> doubleDataFilter{22.0, 10.0};
    for (auto &&value : dataDouble)
    {
        std::cout << doubleDataFilter.RunFilter(value) << ", ";
    }
    std::cout << std::endl;

    std::cout << "------------------------------\n";
    std::vector<int> dataInteger{42, 42, 54, 42, 42, 62, 42};

    LimitedAmplitudeFilter<int> integerDataFilter{42, 5};
    for (auto &&value : dataInteger)
    {
        std::cout << integerDataFilter.RunFilter(value) << ", ";
    }
    std::cout << std::endl;

    return 0;
}
