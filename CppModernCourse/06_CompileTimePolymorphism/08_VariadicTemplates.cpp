/**
 * @file 08_VariadicTemplates.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <iostream>

/**
 * @brief 可变参数模板 Variadic Templates
 * 有时模板必须接受长度不确定的参数列表, 随后编译器在模板实例化时需要知道这些参数,
 * 但想避免针对不同数量的参数编写不同的模板, 这就是可变参数模板存在的理由,
 * 可变参数模板接受数量可变的参数.
 * 
 * ===== 具有特殊语法的末位模板参数来表示可变参数模板, 即 typename... arguments
 * 省略号表示 arguments 是参数包类型, 这意味着可以在模板中声明参数包,
 * 参数包是模板参数, 可以接受零个或多个函数参数.
 * https://en.cppreference.com/w/cpp/language/parameter_pack
 * https://en.cppreference.com/w/cpp/language/variadic_arguments
 * 
 * 标准库在 ＜utility＞ 头文件中包含了一个叫作 std::forward 的函数,
 * 它将检测 arguments 是左值还是右值, 并根据结果执行复制或移动操作.
 * 
 */

// 边界条件
double accumulateVector()
{
    return 0;
}

template<typename DataType, typename... Args>
double accumulateVector(const DataType &first, Args... args)
{
    double res = first + accumulateVector(args...);
    return res;
}

double Sum() // 边界条件
{
    return 0;
}

template<typename T1, typename... T2>
double Sum(T1 p, T2... arg)
{
    double ret = p + Sum(arg...);
    return ret;
}

// 用来终止递归并打印最后一个元素的函数
// 此函数必须在可变参数版本的print定义之前声明（否则将出现neither visible nor found by argument-dependent lookup错误）
template<typename T>
std::ostream &print(std::ostream &os, const T &t)
{
    return os << t; // 包中最后一个元素
}

//包中除最后一个元素之外的其他元素都会调用这个版本的print
template<typename T, typename... Args>
std::ostream &print(std::ostream &os, const T &t, const Args &...rest)
{
    os << t << ",";            // 打印第一个实参，包中元素减一
    return print(os, rest...); // 递归调用，打印剩余实参
}

// -----------------------------------
int main(int argc, const char **argv)
{
    double res = 0.;
    res        = accumulateVector(res, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    printf("the sum value: %f\n", res);

    printf("the sum: %f\n", Sum(2, 2, 2, 2));

    /**
     * @brief Advanced Template Topics 高级模板主题
     * 事实证明模板也被广泛用于高级场合, 特别是在实现库、高性能程序和嵌入式系统固件.
     * Step 1. Template Specialization 模板特化;
     * Step 2. Name Binding 名字绑定;
     * Step 3. Type Function 类型函数;
     * Step 4. Template Meta-programming 模板元编程;
     * 
     * ==== 模板源代码组织 Template Source Code Organization
     * 每次实例化模板时, 编译器必须能够生成使用模板所需的所有代码, 
     * 这意味着所有实例化自定义类或函数的信息必须与模板实例化在同一个编译单元中.
     * 到目前为止, 最流行的方式是完全在头文件中实现模板,
     * 这种方法有一些不方便的地方, 编译时间会增加, 因为具有相同参数的模板可能会被多次实例化;
     * 它还降低了隐藏实现细节的能力, 幸运的是, 泛型编程的好处远远超过了这些不便之处
     * *无论如何, 主流的编译器都会尝试优化编译时间和代码重复的问题.
     * 头文件模板的优点如下:
     * 1. 别人很容易使用代码: 只是对一些头文件应用#include,而不是编译库确保产生的对象文件对链接器可见.
     * 2. 对于编译器来说, 内联头文件模板是非常简单的, 这可以使代码在运行时更快;
     * 3. 当所有的源代码都可用时, 编译器一般可以更好地优化代码.
     * 
     * ==== 运行时多态与编译时多态对比 Polymorphism at Runtime vs. Compile Time
     * 当想要多态时, 应该使用模板; 但有时无法使用模板, 因为在运行前不知道代码使用的类型.
     * *请记住只有当将模板的参数与类型配对时, 才会发生模板实例化.
     * 除非程序正在运行中, 至少在编译时执行这样的配对很烦琐,这种情况下可以使用运行时多态.
     * *模板是实现编译时多态的机制, 实现运行时多态的机制是接口.
     * 
     */

    return 0;
}
