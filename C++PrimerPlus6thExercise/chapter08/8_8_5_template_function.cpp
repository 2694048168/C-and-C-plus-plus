/**
 * @file 8_8_5_template_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <vector>

template<typename T>
// inline T max5(T *arr)
inline T max5(T arr[])
{
    // return *std::max_element(std::begin(arr), std::end(arr));
    return *std::max_element(arr, arr + 5);
}

/**
 * @brief 编写C++程序, 利用模板函数, 返回泛型数组中最大元素
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int size = 5;

    // ----------------------------
    int arr_int[size] = {1, 2, 3, 4, 5};
    std::cout << "The max value of array: " << max5(arr_int) << "\n";

    double arr_double[size] = {1.1, 2.2, 33.3, 43.1, 5.0};
    std::cout << "The max value of array: " << max5(arr_double) << "\n";

    // ----------------------------
    // TODO 如何兼容 C-style 数组, std::array, std::vector, 三种类型数组
    // 模板重载 template overloading?
    // std::array<int, size> array_int{1, 2, 3, 4, 5};
    // std::vector<int> vec_int{1, 2, 3, 4, 5};

    return 0;
}