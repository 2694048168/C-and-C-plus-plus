/**
 * @file test_a.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __A_HPP__
#define __A_HPP__

#include "test/test_b.hpp"

#include <iostream>
#include <memory> /* shared_ptr */

// shared_ptr 循环引用的问题测试
class A
{
public:
    A() = default;

    ~A()
    {
        std::cout << "A is deleted\n";
    }

    std::shared_ptr<B> m_b;
};

#endif // !__A_HPP__