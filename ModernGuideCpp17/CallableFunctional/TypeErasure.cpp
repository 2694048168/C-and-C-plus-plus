/**
 * @file TypeErasure.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-04-11
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ TypeErasure.cpp -std=c++20
 * clang++ TypeErasure.cpp -std=c++20
 * 
 */

#include <iostream>
#include <memory>

// 1. 抽象接口类
template<typename R, typename... Args>
struct ICallable
{
    // 定义通用的函数行为 隐藏真实的类型
    virtual R invoke(Args &&...args) = 0;

    virtual ~ICallable() {}
};

// 2. 桥接类
template<typename T, typename R, typename... Args>
class ICallableImpl : public ICallable<R, Args...>
{
    T callable;

public:
    ICallableImpl(T &&c)
        : callable(std::move(c))
    {
    }

    R invoke(Args &&...args) override
    {
        return callable(std::forward<Args>(args)...);
    }
};

// 3. 函数签名
template<typename Signature>
class MyFunction;

// 4. 具体实现
template<typename R, typename... Args>
class MyFunction<R(Args...)>
{
    std::unique_ptr<ICallable<R, Args...>> funcPtr;

public:
    template<typename T>
    MyFunction(T &&callable)
    {
        funcPtr = std::make_unique<ICallableImpl<T, R, Args...>>(std::forward<T>(callable));
    }

    // overload operator()
    R operator()(Args... args) const
    {
        return funcPtr->invoke(std::forward<Args>(args)...);
    }
};

// --------------------------------------
void func()
{
    std::cout << "Hello Type TypeErasure via Modern C++\n";
}

int add_num(int num1, int num2)
{
    return num1 + num2;
}

float sub_num(float num1, float num2)
{
    return num1 - num2;
}

// --------------------------------------
int main(int argc, const char *argv[])
{
    MyFunction<void()> f(func);
    f();

    MyFunction<int(int, int)> f1(add_num);
    std::cout << f1(1, 1) << std::endl;

    MyFunction<float(float, float)> f2(sub_num);
    std::cout << f2(3.14, 1.12) << std::endl;

    return 0;
}
