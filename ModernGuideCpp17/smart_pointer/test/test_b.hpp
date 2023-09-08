/**
 * @file test_b.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __B_HPP__
#define __B_HPP__

#include <iostream>
#include <memory> /* shared_ptr */

// 前向申明方式
class A;

// shared_ptr 循环引用的问题测试
class B
{
public:
    B() = default;

    ~B()
    {
        std::cout << "B is deleted\n";
    }

    // std::shared_ptr<A> m_a;
    std::weak_ptr<A> m_a;
};

#endif // !__B_HPP__