/**
 * @file 18_12_1_average_temple.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

template<typename T>
T average_list(std::initializer_list<T> list)
{
    return std::accumulate(list.begin(), list.end(), T{0}) / list.size();
}

/**
 * @brief 编写C++程序, 使用模板函数, 计算均值
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // list of double deduced from list contents
    auto q = average_list({15.4, 10.7, 9.0});
    std::cout << q << std::endl;

    // list of int deduced from list contents
    std::cout << average_list({20, 30, 19, 17, 45, 38}) << std::endl;

    // forced list of double
    auto ad = average_list<double>({'A', 70, 65.33});
    std::cout << ad << std::endl;

    std::cout << average_list({1, 2, 3, 4, 5, 6, 7, 8, 9}) << std::endl;

    return 0;
}