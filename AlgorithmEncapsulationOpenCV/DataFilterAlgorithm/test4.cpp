/**
 * @file test4.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ test4.cpp -std=c++20
 * clang++ test4.cpp -std=c++20
 * 
 */

#include "MedianMeanFilter.hpp"

#include <iostream>
#include <vector>

//  -----------------------------------
int main(int argc, const char *argv[])
{
    std::vector<float> dataVec{3.123f, 3.234f, 3.234f, 3.454f, 3.899f, 3.648f};

    MedianMeanFilter<float> medianMeanFilter;
    medianMeanFilter.SetDataVec(dataVec);
    medianMeanFilter.AddData(3.45f);
    medianMeanFilter.AddData(3.56f);
    std::cout << medianMeanFilter.RunFilter() << "\n";

    return 0;
}
