/**
 * @file test5.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ test5.cpp -std=c++20
 * clang++ test5.cpp -std=c++20
 * 
 */

#include "MeanFilter.hpp"

#include <iostream>
#include <vector>

//  -----------------------------------
int main(int argc, const char *argv[])
{
    std::vector<float> dataVec{3.123f, 3.234f, 3.234f, 3.454f, 3.899f, 3.648f};

    MeanFilter<float> meanFilter;
    meanFilter.SetDataVec(dataVec);
    meanFilter.AddData(3.45f);
    meanFilter.AddData(3.56f);
    std::cout << meanFilter.RunFilter() << "\n";

    return 0;
}
