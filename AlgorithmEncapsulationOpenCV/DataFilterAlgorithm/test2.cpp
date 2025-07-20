/**
 * @file test2.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-20
 * 
 * @copyright Copyright (c) 2025
 *
 * g++ test2.cpp -std=c++20 
 * clang++ test2.cpp -std=c++20 
 * 
 */

#include "LowPassFilter.hpp"

#include <iostream>
#include <vector>

//  -------------------------------------
int main(int argc, const char *argv[])
{
    std::vector<float> dataVec{3.123f, 3.234f, 3.234f, 3.454f, 3.899f, 3.648f};

    LowPassFilter<float> lowpassFilter{3.123f};
    lowpassFilter.SetAmplitude(0.25f);
    for (auto &&value : dataVec)
    {
        std::cout << lowpassFilter.RunFilter(value) << ", ";
    }
    std::cout << std::endl;

    return 0;
}
