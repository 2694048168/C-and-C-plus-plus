/**
 * @file 11_9_4_main_time.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_4_time_class.hpp"

void show_time();

/**
 * @brief 编写C++程序, 测试自定义的 Time 时间类
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    show_time();
    std::cout << "==================================\n\n";

    Time aida(3, 35);
    Time tosca(2, 48);
    Time temp;

    std::cout << "Aida and Tosca:\n";
    std::cout << aida << "; " << tosca << std::endl;

    temp = aida + tosca; // operator+()
    std::cout << "Aida + Tosca: " << temp << std::endl;

    temp = aida * 1.17; // member operator*()
    std::cout << "Aida * 1.17: " << temp << std::endl;
    std::cout << "10.0 * Tosca: " << 10.0 * tosca << std::endl;

    return 0;
}

void show_time()
{
    Time planning;
    Time coding(2, 40);
    Time fixing(5, 55);
    Time total;

    std::cout << "planning time = ";
    planning.Show();

    std::cout << "coding time = ";
    coding.Show();

    std::cout << "fixing time = ";
    fixing.Show();

    total = coding + fixing;
    // operator notation
    std::cout << "coding + fixing = ";
    total.Show();

    Time more_fixing(3, 28);
    std::cout << "more fixing time = ";
    more_fixing.Show();

    total = more_fixing + total;
    // function notation
    std::cout << "more_fixing.operator+(total) = ";
    total.Show();
}