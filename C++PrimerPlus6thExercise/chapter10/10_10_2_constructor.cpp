/**
 * @file 10_10_2_constructor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include "10_10_2_person.hpp" 

#include <iostream>


/**
 * @brief 编写C++程序, 理解class 的构造函数 constructor
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Person one;
    one.Show();
    one.FormalShow();
    std::cout << "----------------------------\n";

    Person two{"Wei"};
    two.Show();
    two.FormalShow();
    std::cout << "----------------------------\n";

    Person three{"Wei", "Li"};
    three.Show();
    three.FormalShow();
    std::cout << "----------------------------\n";

    return 0;
}