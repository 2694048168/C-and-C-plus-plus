/**
 * @file 24_multiReturn.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

/** C++ 分享: 多值返回技巧
 * 指针 | 引用 | Lambda 表达式 | 函数式编程 | std::tie | std::tuple | 结构化绑定 | 模板推导
 * 1. 指针和引用: 通过添加两个变量并使用指针和引用来修改函数接口;缺点很明显,当返回值超过两个时,函数接口会变得冗长;
 * 2. 元组Tuple + 解包绑定Tie: 
 *   std::tuple 和 std::tie 的组合提供了一种简洁的方式, 可从函数中返回多个值;
 *  通过使用 std::tie, 可以将元组的元素解包到不同的变量中, 从而提高代码的清晰度;
 * 3. 结构化绑定: C++17 引入了结构化绑定, 进一步简化了多值返回的代码;使用 auto 和结构化绑定能让代码更加直观;
 * 4. 函数回调: 通过传递一个处理返回值的回调函数,用户可以自定义处理方式,使代码结构更加灵活;
 *   这在异步编程和事件处理等场景中特别有用.
 * 5. 自定义输出封装: 自定义输出封装是将输出参数封装在一个结构体中的方法;
 *   这种方法提高了代码的可读性, 尤其适用于需要返回多个值的函数;
 * 6. 模板推导: C++ 的模板推导为开发者提供了一种更灵活、简洁的代码编写方式;
 *   通过模板推导, 可以处理不同的数据类型, 而无需显式指定它们; 
 * 
 */

#include <functional>
#include <iostream>
#include <tuple>

void divideWithReferences(int dividend, int divisor, int &quotient, int &remainder)
{
    quotient  = dividend / divisor;
    remainder = dividend % divisor;
}

void divideWithPointers(int dividend, int divisor, int *quotient, int *remainder)
{
    if (quotient)
        *quotient = dividend / divisor;
    if (remainder)
        *remainder = dividend % divisor;
}

std::tuple<int, int> divide(int dividend, int divisor)
{
    return std::make_tuple(dividend / divisor, dividend % divisor);
}

auto divideStructBind(int dividend, int divisor)
{
    struct result
    {
        int quotient;
        int remainder;
    };

    return result{dividend / divisor, dividend % divisor};
}

void divideCallback(int dividend, int divisor, std::function<void(int, int)> callback)
{
    callback(dividend / divisor, dividend % divisor);
}

// ============= 自定义输出封装 =============
template<class T>
struct out
{
    std::function<void(T)> target;

    out(T *t)
        : target(
              [t](T &&in)
              {
                  if (t)
                      *t = std::move(in);
              })
    {
    }

    template<class... Args>
    void emplace(Args &&...args)
    {
        target(T(std::forward<Args>(args)...));
    }

    template<class X>
    void operator=(X &&x)
    {
        emplace(std::forward<X>(x));
    }

    template<class... Args>
    void operator()(Args &&...args)
    {
        emplace(std::forward<Args>(args)...);
    }
};

void divideCustomOutput(int dividend, int divisor, out<int> &quotient_out, out<int> &remainder_out)
{
    quotient_out.emplace(dividend / divisor);
    remainder_out.emplace(dividend % divisor);
}

// ============= 模板推导 =============
template<typename T1, typename T2>
struct many
{
    T1 quotient;
    T2 remainder;
};

template<class T1, class T2>
many(T1, T2) -> many<T1, T2>;

many<int, int> divideTemplate(int dividend, int divisor)
{
    return many{
        dividend / divisor,
        dividend % divisor,
    };
}

int main(int /* argc */, const char * /* argv[] */)
{
    int quotient, remainder;
    std::tie(quotient, remainder) = divide(14, 3);
    std::cout << quotient << ", " << remainder << std::endl;

    auto [quotient_, remainder_] = divideStructBind(14, 3);
    std::cout << quotient_ << ", " << remainder_ << std::endl;

    divideCallback(14, 3,
                   [](int quotient_, int remainder_) { std::cout << quotient_ << ", " << remainder_ << std::endl; });

    auto [quotient__, remainder__] = divideTemplate(14, 3);
    std::cout << quotient__ << ", " << remainder__ << std::endl;

    return 0;
}
