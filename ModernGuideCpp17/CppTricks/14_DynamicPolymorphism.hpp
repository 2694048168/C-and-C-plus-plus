/**
 * @file 14_DynamicPolymorphism.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <string>

/**
 * @brief 假设当前需要设计一个 Info 类中包含 getName() 接口,
 * 有不同的子类对该接口进行实现, 很容易想到的方法是利用 C++ 的多态机制;
 * 传统动态多态的实现方法, 在运行时通过查询虚函数表, 找到实际调用接口, 返回正确的类名
 *
 */
struct IInfo
{
    [[nodiscard]] virtual std::string getClassName() = 0;
};

class A : public IInfo
{
    [[nodiscard]] std::string getClassName() override
    {
        return "A";
    }
};

class B : public IInfo
{
    [[nodiscard]] std::string getClassName() override
    {
        return "B";
    }
};
