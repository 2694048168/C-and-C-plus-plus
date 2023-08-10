/**
 * @file 11_9_5_main_stone.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_5_stone_weight.hpp"

#include <iostream>

/**
 * @brief 编写C++程序, 测试自定义类 StoneWeight 的功能
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    StoneWeight obj1;
    std::cout << "The object1 info: " << obj1 << "\n";

    StoneWeight obj2(66.6);
    std::cout << "The object2 info: " << obj2 << "\n";
    obj2.set_status(StoneWeight::floating_point_pounds_form);
    std::cout << "The object2 info: " << obj2 << "\n";

    StoneWeight obj3(42, 8.8);
    std::cout << "The object3 info: " << obj3 << "\n";
    obj2.set_status(StoneWeight::integer_pounds_form);
    std::cout << "The object3 info: " << obj3 << "\n";

    // operator overloading
    std::cout << "The object add(1+2): " << obj1 + obj2 << std::endl;
    std::cout << "The object add(2+3): " << obj2 + obj3 << std::endl;

    std::cout << "The object sub(2-3): " << obj2 - obj3 << std::endl;

    // TODO maybe some letter issue for operator '*'?
    std::cout << "The object mul(3*obj3): " << 3 * obj3 << std::endl;
    std::cout << "The object add(obj1*2): " << obj1 * 2 << std::endl;

    return 0;
}