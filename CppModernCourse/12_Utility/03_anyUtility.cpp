/**
 * @file 03_anyUtility.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <any>
#include <cassert>
#include <cstdio>

/**
 * @brief any
 * any 是存储任意类型的单个值的类, 它不是类模板, 要将 any 转换为具体类型, 请使用 any 强制转换(cast),
 * 它是一个非成员函数模板, 任何强制转换都是类型安全的;
 * 如果尝试强制转换 any 类型并且类型不匹配, 则会出现异常.
 * *使用 any, 可以在没有模板的情况下执行某些类型的泛型编程.
 * 
 * *stdlib 的＜any＞头文件中有 std::any
 * 
 */

struct EscapeCapsule
{
    EscapeCapsule(int x)
        : weight_kg{x}
    {
    }

    int weight_kg;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 要将值存储到 any 中, 请使用 emplace 方法模板,
     * 它接受单个模板参数, 该参数就是要存储到 any 中的类型(存储类型).
     * 传递给 emplace 的任何参数都会被转发给给定存储类型的适当的构造函数.
     * 要提取值, 可以使用 any_cast, 它接受与 any 的当前存储类型相对应的模板参数(称为 any 的状态).
     * 将 any 作为唯一参数传递给 any_cast, 只要 any 的状态与模板参数匹配, 就会得到所需的类型,
     * 如果状态不匹配, 则会得到 bad_any_cast 异常.
     * 
     */
    printf("std::any allows us to std::any_cast into a type\n");
    std::any hag;
    hag.emplace<EscapeCapsule>(600);

    auto capsule = std::any_cast<EscapeCapsule>(hag);
    assert(capsule.weight_kg == 600);

    try
    {
        auto ret = std::any_cast<float>(hag);
        printf("the ret value: %f", ret);
    }
    catch (const std::bad_any_cast &exp)
    {
        printf("the throw exception: std::bad_any_cast, %s\n", exp.what());
    }
    catch (const std::exception &exp)
    {
        printf("the throw exception %s\n", exp.what());
    }

    return 0;
}
