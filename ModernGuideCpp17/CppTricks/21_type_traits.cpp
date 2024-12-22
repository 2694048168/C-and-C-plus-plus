/**
 * @file 21_type_traits.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

/**
 * @brief <type_traits> header-file 提供一系列的类模板和函数,
 * 可用于编译时期的类型分析和转换,可以确保模板类型的安全和正确使用,以及实现条件编译
 * 
 */

template<typename T>
void testIntegral(T num)
{
    if (std::is_integral<T>::value)
    {
        std::cout << std::boolalpha << num << " IS an integral type\n";
    }
    else
    {
        std::cout << std::boolalpha << num << " is NOT an integral type\n";
    }
}

class IMeasurable
{
public:
    [[nodiscard]] virtual double length() noexcept = 0;
};

class Complex final : public IMeasurable
{
public:
    Complex(double r, double i)
        : imaginary{i}
        , real{r}
    {
    }

    [[nodiscard]] double length() noexcept override
    {
        return std::sqrt(imaginary * imaginary + real * real);
    }

private:
    double imaginary = 0.;
    double real      = 0.;
};

// if constexpr since C++17 + type_traits 中的模板变量都是常量
// 编译时期的条件语句(常量表达式), 在编译时期对条件表达式进行判断,
// 当条件表达式的分支为 true, 对应分支代码才会被编译; 否则对应分支代码会被舍弃
template<typename T>
double length(T &t)
{
    // 判断参数类型是不是数值类型,返回绝对值
    if constexpr (std::is_arithmetic_v<T>)
    {
        if (t < 0)
            return -t;
        return t;
    }
    // 判断T是不是继承了IMeasurable的类接口,调用对应的函数
    else if constexpr (std::is_base_of_v<IMeasurable, T>)
    {
        return t.length();
    }
    // 都不满足则直接返回 zero
    else
    {
        return 0;
    }
}

// static_assert 用于在编译阶段判断条件是否为真
// 如果判断条件为 false, 则会停止编译, 报错静态断言错误
template<typename T>
double length_(T &t)
{
    // 替代了 else 分支的需求
    static_assert(std::is_arithmetic_v<T> || std::is_base_of_v<IMeasurable, T>,
                  "Error: 使用了不支持计算长度(length)的类型");

    // 判断参数类型是不是数值类型,返回绝对值
    if constexpr (std::is_arithmetic_v<T>)
    {
        if (t < 0)
            return -t;
        return t;
    }
    // 判断T是不是继承了IMeasurable的类接口,调用对应的函数
    else if constexpr (std::is_base_of_v<IMeasurable, T>)
    {
        return t.length();
    }
}

int main(int argc, const char *argv[])
{
    testIntegral(3);
    testIntegral(false);
    testIntegral('A');
    testIntegral(3.5);
    testIntegral("ABC");
    testIntegral(30L);

    std::cout << "---------------------\n";
    double val = -13.223;
    std::cout << length(val) << '\n';

    Complex v = {1.5, 3.0};
    std::cout << length(v) << '\n';

    std::vector<int> num_vec{1, 3, 5};
    // std::cout << length_(num_vec) << '\n';
    std::cout << length_(num_vec[1]) << '\n';

    return 0;
}
