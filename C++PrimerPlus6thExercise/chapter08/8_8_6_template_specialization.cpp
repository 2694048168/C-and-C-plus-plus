/**
 * @file 8_8_6_template_specialization.cpp
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
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

template<typename T>
inline T max_n(T arr[], const unsigned num)
{
    return *std::max_element(arr, arr + num);
}

template<typename T>
inline T max_n(const std::vector<T> &arr_vec, const unsigned num)
{
    return *std::max_element(arr_vec.begin(), arr_vec.end());
}

// explicit specialization for the "char*" array type
template<>
char *max_n<char *>(char *arr[], const unsigned num)
{
    unsigned max_size = 0;
    unsigned max_idx  = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (strlen(arr[i]) > max_size)
        {
            max_size = strlen(arr[i]);
            max_idx  = i;
        }
    }

    return arr[max_idx];
}

/**
 * @brief 编写C++程序, 利用模板函数, 返回泛型数组中最大元素, 并显示具体化该模板函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ----------------------------
    int arr_int[6] = {1, 2, 3, 4, 5, 42};
    std::cout << "The max value of array: " << max_n(arr_int, 6) << "\n";

    double arr_double[4] = {1.1, 42.2, 33.3, 13.1};
    std::cout << "The max value of array: " << max_n(arr_double, 4) << "\n";

    char *arr_char[5] = {(char *)"Hello", (char *)"CPP", (char *)"World", (char *)"Wei", (char *)"Li"};
    std::cout << "The max size of string: " << max_n(arr_char, 5) << "\n";

    // ----------------------------
    // 模板重载
    std::vector<int> vec_int{7, 3, 4, 5, 1};
    std::cout << "The max value of array: " << max_n(vec_int, 5) << "\n";

    std::vector<float> vec_float{7.2f, 3.1f, 4.4f, 59.1f, 1.f};
    std::cout << "The max value of array: " << max_n(vec_float, 5) << "\n";

    return 0;
}