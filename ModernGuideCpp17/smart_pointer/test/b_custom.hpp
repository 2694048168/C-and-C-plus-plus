/**
 * @file b_custom.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __B_CUSTOM_HPP__
#define __B_CUSTOM_HPP__

#include "utility/weak_ptr.hpp"
using namespace WeiLi::utility;

#include <iostream>

// 前向申明方式
class A_CUSTOM;

// shared_ptr 循环引用的问题测试
class B_CUSTOM
{
public:
    B_CUSTOM() = default;

    ~B_CUSTOM()
    {
        std::cout << "B_CUSTOM is deleted\n";
    }

    WeakPtr<A_CUSTOM> m_a;
};

#endif // !__B_CUSTOM_HPP__