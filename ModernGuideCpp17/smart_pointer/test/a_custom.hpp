/**
 * @file a_custom.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __A_CUSTOM_HPP__
#define __A_CUSTOM_HPP__

#include "test/b_custom.hpp"
#include "utility/shared_ptr.hpp"
using namespace WeiLi::utility;

#include <iostream>

// shared_ptr 循环引用的问题测试
class A_CUSTOM
{
public:
    A_CUSTOM() = default;

    ~A_CUSTOM()
    {
        std::cout << "A_CUSTOM is deleted\n";
    }

    SharedPtr<B_CUSTOM> m_b;
};

#endif // !__A_CUSTOM_HPP__