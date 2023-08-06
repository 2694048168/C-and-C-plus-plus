/**
 * @file 10_10_8_main_adt.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "10_10_8_list_adt.hpp"

#include <iostream>

void add_2(Item &item)
{
    item += 2;
}

void mul_2(Item &item)
{
    item *= 2;
}

/**
 * @brief 编写C++程序, 抽象数据类型(Abstract Data Type, ADT)
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    ListADT list_1;
    list_1.show_item();

    if (list_1.is_empty())
    {
        std::cout << "The list is empty\n";
    }
    else
    {
        std::cout << "The list is NOT empty\n";
    }

    if (list_1.is_full())
    {
        std::cout << "The list is full\n";
    }
    else
    {
        std::cout << "The list is NOT full\n";
    }

    list_1.list_modify(24);
    list_1.list_modify(42, 6);
    list_1.show_item();

    if (list_1.is_empty())
    {
        std::cout << "The list is empty\n";
    }
    else
    {
        std::cout << "The list is NOT empty\n";
    }

    if (list_1.is_full())
    {
        std::cout << "The list is full\n";
    }
    else
    {
        std::cout << "The list is NOT full\n";
    }

    // --------------------------
    const unsigned size = 10;

    int arr[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ListADT list_3(arr, size);
    list_3.show_item();

    if (list_3.is_empty())
    {
        std::cout << "The list is empty\n";
    }
    else
    {
        std::cout << "The list is NOT empty\n";
    }

    if (list_3.is_full())
    {
        std::cout << "The list is full\n";
    }
    else
    {
        std::cout << "The list is NOT full\n";
    }

    // ------------------------------
    // 针对元素做单一运算符, 一元运算操作
    list_3.item_op(3, add_2);
    list_3.item_op(add_2);
    list_3.show_item();

    list_3.item_op(3, mul_2);
    list_3.item_op(mul_2);
    list_3.show_item();

    return 0;
}