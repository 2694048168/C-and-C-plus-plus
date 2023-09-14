/**
 * @file 18_12_3_variadic_template.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

template<typename... Args>
long double sum_values(Args... args)
{
    return (... + args);
}

/**
 * @brief 编写C++程序, 可变参数模板函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "the sum: " << sum_values(1, 2.3, 3.14f) << std::endl;

    std::cout << "the sum: " << sum_values(1, 2, 3) << std::endl;

    std::cout << "the sum: " << sum_values(1.2, 2.3, 3.4) << std::endl;

    std::cout << "the sum: " << sum_values(1.21f, 2.33f, 3.41f) << std::endl;

    return 0;
}