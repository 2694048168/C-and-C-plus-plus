/**
 * @file 7_13_6_array_operator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

unsigned int fill_array(double *arr, const unsigned int size);

void show_array(const double *arr, const unsigned int size);

void reverse_array(double *arr, const unsigned int size);
void reverse_arr(double *arr, const unsigned int size);

/**
 * @brief 编写C++程序, 填充数组, 显示数组, 反转数组等功能函数操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int size = 6;

    double arr[size] = {0};

    std::cout << "=============================\n";
    unsigned num_element = fill_array(arr, size);
    std::cout << "The number fo array is: " << num_element << "\n";

    unsigned num_size = std::min(num_element, size);

    std::cout << "=============================\n";
    std::cout << "The element value of array: \n";
    show_array(arr, num_size);

    reverse_array(arr, num_size);

    std::cout << "=============================\n";
    std::cout << "Reverse array and value of array: \n";
    show_array(arr, num_size);

    reverse_arr(arr, num_size);

    std::cout << "=============================\n";
    std::cout << "Reverse array and value of array: \n";
    show_array(arr, num_size);

    return 0;
}

unsigned int fill_array(double *arr, const unsigned int size)
{
    unsigned int num_elem = 0;
    std::cout << "Please enter number: ";
    double input;
    while (std::cin >> input && num_elem < size)
    {
        arr[num_elem] = input;
        ++num_elem;
    }

    return num_elem;
}

// size == std::min(size, num_elem)
void show_array(const double *arr, const unsigned int size)
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// size == std::min(size, num_elem)
void reverse_array(double *arr, const unsigned int size)
{
    // 双指针策略
    double *head = arr;
    double *tail = arr + size - 1;
    for (int i = 0; i < size / 2; ++i)
    {
        if (head < tail)
        {
            double tmp = *tail;
            *tail      = *head;
            *head      = tmp;

            ++head;
            --tail;
        }
    }
}

// size == std::min(size, num_elem)
void reverse_arr(double *arr, const unsigned int size)
{
    // 双指针策略,
    double *head = arr;
    ++head;
    double *tail = arr + size - 1;
    --tail;
    for (int i = 0; i < size / 2; ++i)
    {
        if (head < tail)
        {
            double tmp = *tail;
            *tail      = *head;
            *head      = tmp;

            ++head;
            --tail;
        }
    }
}