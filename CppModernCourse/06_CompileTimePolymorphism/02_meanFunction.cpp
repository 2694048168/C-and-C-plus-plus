/**
 * @file 02_meanFunction.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>

/**
 * @brief mean 该函数使用求和运算和除法运算计算 double 类型数组的均值.
 * 1. mean 支持其他数值类型, 函数重载的范式;
 * 2. mean 支持其他数值类型, 函数模板的范式;
 * 
 * *Template Type Deduction 模板类型扣除
 * 要解决这个复制粘贴问题, 需要用到泛型编程, 在这种编程风格下, 可以用尚未确定的类型进行编程;
 * 利用 C++对模板的支持, 便可以实现泛型编程, 模板允许编译器根据使用中的类型实例化自定义的类或函数.
 * 
 */
double mean(const double *values, size_t length)
{
    double result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}

float mean(const float *values, size_t length)
{
    float result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}

int mean(const int *values, size_t length)
{
    int result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}

// Template Type Deduction 模板类型扣除
template<typename T>
T mean_func(T *values, size_t length)
{
    T result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}

// ----------------------------------
int main(int argc, const char **argv)
{
    printf("========the function overload=======\n");
    const double nums_d[]{1.0, 2.0, 3.0, 4.0};
    const auto   result1 = mean(nums_d, 4);
    printf("double: %f\n", result1);

    float      nums_f[]{1.0f, 2.0f, 3.0f, 4.0f};
    const auto result2 = mean(nums_f, 4);
    printf("float: %f\n", result2);

    int        nums_c[]{1, 2, 3, 4};
    const auto result3 = mean(nums_c, 4);
    printf("int: %d\n", result3);

    printf("========the function template=======\n");
    double     nums_double[]{1.0, 2.0, 3.0, 4.0};
    const auto result1_ = mean_func<double>(nums_double, 4);
    printf("double: %f\n", result1_);

    const auto result2_ = mean_func(nums_f, 4);
    printf("float: %f\n", result2_);

    const auto result3_ = mean_func(nums_c, 4);
    printf("int: %d\n", result3_);

    size_t     nums_unsigned[]{1, 2, 3, 4};
    const auto result_ = mean_func<size_t>(nums_unsigned, 4);
    printf("size_t: %zd\n", result_);

    // 幸运的是, 当调用模板函数时, 一般可以省略模板参数;
    // 编译器判断正确模板参数的过程叫作模板类型推断.
    // 有时模板参数无法被推导出来,例如如果模板函数的返回类型是模板参数,则必须明确地指定模板参数.
    // https://en.cppreference.com/w/cpp/language/templates
    // Template Type Deduction
    printf("========Template Type Deduction=======\n");
    size_t nums_[]{1, 2, 3, 4};
    const auto   res = mean_func(nums_, 4);
    printf("size_t: %zd\n", res);

    return 0;
}
