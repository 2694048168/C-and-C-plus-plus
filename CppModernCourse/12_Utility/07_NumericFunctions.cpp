/**
 * @file 07_NumericFunctions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <complex>
#include <cstdio>
#include <random>
#include <ratio>

/**
 * @brief Numerics 数值
 * ?如何使用常见的数学函数和常数处理数值,以及如何处理复数,生成随机数,数值限制,数值转换并计算比率.
 * 
 * ====数值函数 Numeric Functions
 * *stdlib Numerics 库和 Boost Math 库提供了大量的数值/数学函数
 * https://en.cppreference.com/w/cpp/numeric
 * https://en.cppreference.com/w/cpp/numeric/math
 * 
 * ====复数 Complex Numbers
 * *虚数在控制理论,流体动力学,电气工程,信号分析,数论和量子物理等领域都有应用.
 * stdlib 在＜complex＞头文件中提供了 std::complex 类模板,
 * 它接受实部和虚部的基础类型的模板参数, 此模板参数必须是基本浮点类型之一.
 * 非成员函数 std::real 和 std::imag 可以提取复数的实部和虚部.
 * 
 * ====数学常数 Mathematical Constants
 * ====随机数 Random Numbers
 * 
 * 
 * 
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::complex has a real and imaginary component\n");
    std::complex<double> a{0.5, 14.13};
    printf("the real of complex: %f\n", std::real(a));
    printf("the imag of complex: %f\n", std::imag(a));

    /**
     * @brief 在某些情况下, 需要生成随机数, 在科学计算中可能需要基于随机数运行大量模拟,
     * 这些数字需要模拟具有某些特征的随机过程, 例如泊松分布或正态分布.
     * *通常希望这些模拟是可重复的, 因此负责生成随机性的代码(随机数引擎)应该在给定相同输入的情况下产生相同的输出.
     * *这种随机数引擎有时被称为伪随机数引擎.
     *
     * 在密码学中可能需要用随机数来保护信息, 在这种情况下,几乎不可能有人获得类似的随机数流;
     * 意外使用伪随机数引擎通常会严重损害原本安全的密码系统.
     * 
     * ?如果只是想要使用随机数, 那么只需使用 stdlib＜random＞头文件
     * 
     * ==== 随机数引擎 Random Number Engines
     * 随机数引擎生成随机比特, Boost 和 stdlib 有很多令人眼花缭乱的随机数引擎.
     * 有一个通用规则: 
     * *如果需要可重复的伪随机数, 请考虑使用 Mersenne Twister 引擎std::mt19937_64;
     * *如果需要加密安全的随机数, 请考虑使用 std::random_device.
     * Mersenne Twister 具有一些理想的模拟统计特性, 为其构造函数提供一个整数种子值,
     * 便可以完全确定随机数的序列.
     * *所有随机数引擎都是函数对象, 因此要获取随机数, 请使用 operator()
     * 
     * =====随机数分布 Random Number Distributions
     * *随机数分布是将数字映射到概率密度的数学函数; 离散分布和连续分布;
     * 随机数分布列表 A Partial List of Random Number Distributions
     * 
     */
    printf("\nmt19937_64 is pseudorandom\n");
    std::mt19937_64 mt_engine{91586};

    // 因为它是一个伪随机数引擎, 所以可以保证每次都能获得相同的随机数序列,
    // TODO:这个序列完全由种子决定
    assert(mt_engine() == 8346843996631475880);
    assert(mt_engine() == 2237671392849523263);
    assert(mt_engine() == 7333164488732543658);

    printf("\nstd::random_device is invocable\n");
    std::random_device rd_engine{};
    printf("the value of random: %d\n", rd_engine());

    printf("\nstd::uniform_int_distribution produces uniform ints\n");
    std::uniform_int_distribution<int> int_d{0, 10};

    const size_t n{1'000'000};
    int          sum{};
    for (size_t i{}; i < n; i++)
    {
        sum += int_d(mt_engine);
    }
    const auto sample_mean = sum / double{n};
    printf("sample_mean == %f\n", sample_mean);

    /**
     * @brief Numeric Limits 
     * stdlib 在＜limits＞头文件中提供了类模板 std::numeric_limits,
     *  *以便提供有关算术类型的各种属性的编译期信息;
     * 例如, 如果要识别给定类型 T 的最小有限值, 可以使用静态成员函数 std::numeric_limits＜T＞::min()
     * 
     * ?Boost 提供了 Numeric Conversion 库, 其中包含一组用于在数值对象之间进行转换的工具
     */
    printf("\nstd::numeric_limits::min provides the smallest finite value.\n");
    auto my_cup    = std::numeric_limits<int>::min();
    auto underflow = my_cup - 1;
    assert(my_cup < underflow);
    printf("the limits min of INT: %d\n", my_cup);

    /**
     * @brief 编译时有理数算术 Compile-Time Rational Arithmetic
     * stdlib ＜ratio＞头文件中的 std::ratio 是一个类模板,使用它能够在编译时进行有理数计算;
     * 这需向 std::ratio 提供两个模板参数,分别作为分子和分母;
     * 这定义了一种可用于计算有理数表达式的新类型.
     * 
     * *使用 std::ratio 执行编译时计算的方式是使用模板元编程技术.
     * ?在编译时进行计算当然总是比在运行时进行计算要好, 这样程序的效率更高, 因为它们在运行时需要做的计算更少.
     */
    printf("\n=========std::ratio=========\n");
    using ten        = std::ratio<10, 1>;
    using two_thirds = std::ratio<2, 3>;

    using result = std::ratio_multiply<ten, two_thirds>;
    assert(result::num == 20);
    assert(result::den == 3);
    printf("the num: %lld, and the den: %lld\n", result::num, result::den);

    return 0;
}
