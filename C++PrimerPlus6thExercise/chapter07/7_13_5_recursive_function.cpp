/**
 * @file 7_13_5_recursive_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

void compute_recursive(const unsigned int &num, unsigned int &result)
{
    if (num == 0 || num == 1)
    {
        result = 1;
    }
    else
    {
        unsigned result_ = 0;
        compute_recursive(num - 1, result_);
        result = num * result_;
    }
}

/**
 * @brief 编写C++程序, 利用递归函数计算阶乘并同时显示计算的结果
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter an integer to compute: ";

    unsigned int num    = 0;
    unsigned int result = 0;

    while (std::cin >> num)
    {
        compute_recursive(num, result);
        std::cout << "==== The result is: " << result << std::endl;

        std::cout << "Next integer (q to quit): ";
    }
    std::cout << "bye\n";

    return 0;
}