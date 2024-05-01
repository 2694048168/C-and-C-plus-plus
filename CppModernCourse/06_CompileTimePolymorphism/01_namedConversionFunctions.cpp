/**
 * @file 01_namedConversionFunctions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>
#include <stdexcept>

/**
 * @brief Named Conversion Functions 类型转换函数
 * 类型转换是将一种类型显式转换为另一种类型的语言特性.
 * 在不能使用隐式转换或构造函数来得到所需类型的情况下, 可以用类型转换函数.
 * 所有类型转换函数都接受一个单一的对象参数, 即要被转换的对象 object-to-cast,
 * 以及一个单一的类型参数, 即转换后的类型 desired-type.
 * *named-conversion＜desired-type＞(object-to-cast)
 * 1. 如果需要修改一个 const 对象, 那么首先需要去除 const 修饰符,
 *    类型转换函数 const_cast 允许执行这个操作;
 * 2.反转隐式转换(static_cast)或者用不同的类型重新解释内存(reinterpret_cast).
 * NOTE: 虽然类型转换函数在技术上不是模板函数, 但它们在概念上非常接近模板—这种关系反映在它们的语法相似性上.
 * 
 * ===== const_cast 函数可以去掉 const 修饰符, 允许修改 const 值;
 * object-to-cast 对象是const 类型的, 而desired-type是那个类型去掉 const 修饰符后的类型;
 * *使用 const_cast 从对象中去除 volatile 修饰符.
 * 
 * ==== static_cast 可以反转定义良好的隐式转换, 比如将一个整数类型转换为另一个整数类型.
 * object-to-cast 是 desired-type 隐式转换成的某个类型,
 * *需要 static_cast 的原因是隐式转换(implicit casts)是不可逆的
 * 
 * ==== reinterpret_cast
 * 在底层编程中, 必须执行非良好定义的类型转换,
 * 在系统编程中, 特别是在嵌入式环境中, 经常需要完全控制读取内存的方式.
 * reinterpret_cast 使你拥有这样的控制权, 但确保这些转换的正确性完全由你负责.
 * *reinterpret_cast 带有一个类型参数—对应于所需的指针类型, 和一个普通参数—对应结果应指向的内存地址.
 * 
 * ==== narrow_cast
 * 一种自定义的 static_cast, 它执行运行时窄化检查.
 * *窄化会导致信息丢失, 想想将 int 转换为 short 的情形,
 * 只要 int 的值适合 short, 转换就是可逆的, 不会发生窄化;
 * 如果 int 的值太大了, 无法转换为 short, 转换就是不可逆的, 结果就是窄化结果.
 * 
 * 
 */
void carbon_thaw(const int &encased_solo)
{
    // encased_solo++; // Compiler error; modifying const
    auto &hibernation_sick_solo = const_cast<int &>(encased_solo);
    hibernation_sick_solo++;
}

// 请注意, short* 到 void* 的隐式转换定义良好
// 尝试使用 static_cast 进行错误的转换, 如将 char* 转换为 float*, 将导致编译错误
short increment_as_short(void *target)
{
    auto as_short = static_cast<short *>(target);
    *as_short     = *as_short + 1;
    return *as_short;
}

// narrow_cast
template<typename To, typename From>
To narrow_cast(From value)
{
    const auto converted = static_cast<To>(value);
    const auto backwards = static_cast<From>(converted);
    if (value != backwards)
        throw std::runtime_error{"Narrowed!"};
    return converted;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("======== const_cast ========\n");
    int val = 41;
    printf("the value of val before: %d\n", val);
    carbon_thaw(val);
    printf("the value of val after: %d\n", val);

    printf("======== static_cast ========\n");
    short beast{665};
    auto  mark_of_the_beast = increment_as_short(&beast);
    printf("%d is the mark_of_the_beast.\n", mark_of_the_beast);

    // float on          = 3.5166666666;
    // auto  not_alright = static_cast＜char *＞(&on); // !ERROR Bang!

    printf("======== narrow_cast ========\n");
    // If the value of int is too big for the short,
    // the conversion isn’t reversible and results in narrowing.
    int perfect{496};

    const auto perfect_short = narrow_cast<short>(perfect);
    printf("perfect_short: %d\n", perfect_short);

    try
    {
        int cyclic{142857};

        const auto cyclic_short = narrow_cast<short>(cyclic);
        printf("cyclic_short: %d\n", cyclic_short);
    }
    catch (const std::runtime_error &e)
    {
        printf("Exception: %s\n", e.what());
    }

    printf("======== reinterpret_cast ========\n");
    try
    {
        // 该程序可以编译, 但是如果 0x1000 不可读取, 则可以预料到运行时程序会崩溃
        auto timer = reinterpret_cast<const unsigned long *>(0x1000);
        printf("Timer is %lu.", *timer);
    }
    catch (...)
    {
        printf("Exception: the address is Illegal.\n");
    }

    return 0;
}
