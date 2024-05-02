/**
 * @file 02_typeAliasing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>
#include <stdexcept>

/**
 * @brief Type Aliasing 类型别名
 * 类型别名定义一个名字, 该名字指代先前定义的名字,
 * 可以将类型别名作为现有类型名称的同义词, 类型与指代它的类型别名没有区别,
 * !类型别名不能更改现有类型名称的含义.
 * using type-alias = type-id;
 * *类型别名作为替代品来简化代码
 * !类型别名可出现在任何作用域（块，类或命名空间）内
 * 
 * *可以将模板参数引入类型别名, 实现了两个重要的用法:
 * 1. 可以在模板参数中执行局部应用(partial application),
 *    局部应用是将一定数量的参数固定, 从而生成具有更少模板参数的另一个模板的过程.
 * 2. 可以使用完全指定的模板参数集为模板定义类型别名.
 * 3. 模板实例化可能非常冗长, 类型别名可以帮助避免腕管综合征.
 * 
 * TODO: 可以简化模板参数中的类型, 有一些很长的类型名, 同时便于后续替换和维护
 * 
 */
template<typename To, typename From>
struct NarrowCaster
{
    To cast(From value) const
    {
        const auto converted = static_cast<To>(value);
        // ?partial application
        const auto backwards = static_cast<From>(converted);

        if (value != backwards)
            throw std::runtime_error{"Narrowed!"};

        return converted;
    }
};

template<typename From>
using short_caster = NarrowCaster<short, From>;

// ----------------------------------
int main(int argc, const char **argv)
{
    try
    {
        const short_caster<int> caster;

        const auto cyclic_short = caster.cast(142857);
        printf("cyclic_short: %d\n", cyclic_short);
    }
    catch (const std::runtime_error &e)
    {
        printf("Exception: %s\n", e.what());
    }

    return 0;
}
