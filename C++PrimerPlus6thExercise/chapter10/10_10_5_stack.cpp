/**
 * @file 10_10_5_stack.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <stack>
#include <string_view>

struct Customer
{
    std::string_view fullname;
    double           payment;
};

/**
 * @brief 编写C++程序, 利用 stack 数据结构存储结构体, 并进行添加和删除
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    static double total_payment = 0.0;

    std::stack<Customer> customer;

    customer.push({"wei li", 3456.8});
    customer.push({"wei", 456.8});
    customer.push({"li", 99056.8});
    customer.push({"Li", 1256.8});
    customer.push({"Wei", 35656.8});

    while (!customer.empty())
    {
        total_payment += customer.top().payment;
        customer.pop();
        std::cout << "The Total Payment: " << total_payment << "\n";
    }

    return 0;
}