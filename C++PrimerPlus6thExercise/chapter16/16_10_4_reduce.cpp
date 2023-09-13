/**
 * @file 16_10_4_reduce.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <iostream>

int reduce_custom(long arr[], int n);

/**
 * @brief 编写C++程序, 对 C-style 数组排序, 删除重复值, 缩减数组中元素数量
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned SIZE = 24;

    long arr[SIZE] = {1, 2, 2, 3, 4, 5, 5, 6, 34, 5, 6, 7, 7, 8, 9, 10, 12, 32, 34, 34, 590, 89, 88, 99};

    auto print = [&arr](int num)
    {
        unsigned long num_element = 0;
        std::cout << "---------------------------\n";
        for (size_t i = 0; i < num; ++i)
        {
            std::cout << arr[i] << " ";
            ++num_element;
        }
        std::cout << "\n==== SIZE of element: " << num_element;
        std::cout << "\n---------------------------\n";
    };

    print(SIZE);

    int flag = reduce_custom(arr, SIZE);
    if (flag)
    {
        print(flag);
    }

    return 0;
}

int reduce_custom(long arr[], int len)
{
    for (size_t i = 0; i < len - 1; i++)
    {
        for (size_t j = 0; j < len - 1 - i; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }

    auto iter = std::unique(arr, arr + len);

    // TODO debug?
    auto idx_ptr = arr;
    int  size    = 0;
    for (size_t idx = 0; idx_ptr != iter; ++idx)
    {
        arr[idx] = arr[idx];
        ++idx_ptr;
        ++size;
    }

    return size;
}