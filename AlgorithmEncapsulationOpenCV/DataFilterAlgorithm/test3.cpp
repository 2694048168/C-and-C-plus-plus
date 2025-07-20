/**
 * @file test3.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-20
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ test3.cpp -std=c++20
 * clang++ test3.cpp -std=c++20
 * 
 */

#include "MedianFilter.hpp"

#include <iostream>
#include <vector>

//  -----------------------------------
int main(int argc, const char *argv[])
{
    std::vector<float> dataVec{3.123f, 3.234f, 3.234f, 3.454f, 3.899f, 3.648f};

    MedianFilter<float> medianFilter;
    medianFilter.SetDataVec(dataVec);
    medianFilter.AddData(3.45f);
    medianFilter.AddData(3.56f);
    std::cout << medianFilter.RunFilter() << "\n";

    return 0;
}
