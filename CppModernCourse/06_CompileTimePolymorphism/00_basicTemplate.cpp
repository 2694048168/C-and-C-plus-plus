/**
 * @file 00_basicTemplate.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief  Templates 模板
 * C++ 通过模板（template）实现了编译时多态, 模板是带有模板参数的类或函数.
 * 这些参数可以代表任何类型, 包括基本类型和用户自定义类型,
 * 当编译器看到模板与类型一起使用时, 就会产出定制的模板实例化.
 * 模板实例化（template instantiation）是指从模板创建类或函数的过程,
 * 令人困惑的是,也可以把"模板实例化"称为模板实例化过程的结果, 模板实例化有时被称为具体类和具体类型.
 * 简单来说, 不需要把常用的代码到处复制粘贴, 而是创造一个模板,
 * 这样当编译器在模板参数中遇到新的类型组合时, 便会生成新的模板实例.
 * 
 * ===== Declaring Templates 用模板前缀来声明模板
 * 模板前缀由关键字 template 和尖括号 ＜ ＞ 组成,
 * 在尖括号内, 可以放置一个或多个模板参数的声明, 
 * 使用关键字 typename 或 class 后跟标识符来声明模板参数
 * 
 * ===== Instantiating Templates 实例化模板
 * *模板名称和参数的组合当作普通的类型来处理: 可以使用任意初始化语法.
 * *事实上, 它被用于一组名为类型转换函数的语言特性中.
 * 
 */
// Template Class Definitions
template<typename X, typename Y, typename Z>
class TemplateClass
{
public:
    ~TemplateClass()
    {
        if (m_pMember)
        {
            delete m_pMember;
            m_pMember = nullptr;
        }
    }

    X func(Y &y)
    {
        if (nullptr == m_pMember)
            m_pMember = new Z{42};
        return y + (*m_pMember);
    }

private:
    Z *m_pMember = nullptr;
};

// Instantiating Templates
// tc_name＜t_param1, t_param2, ...＞ my_concrete_class{ ... };
TemplateClass<int, int, int> my_class{};

// Template Function Definitions
template<typename X_type, typename Y_type, typename Z_type>
X_type template_function(Y_type &arg1, const Z_type *arg2)
{
    X_type x = 0;

    x = arg1 + (*arg2);

    return x;
}

// Instantiating Templates
// auto result = tf_name＜t_param1, t_param2, ...＞(f_param1, f_param2, ...);
auto my_func = template_function<float, float, float>;

// -----------------------------------
int main(int argc, const char **argv)
{
    int  val = 2;
    auto res = my_class.func(val);
    printf("the result of template class: %d\n", res);

    float num1   = .12f;
    float num2   = 2.12f;
    auto  result = my_func(num2, &num1);
    printf("the result of template function: %f\n", result);

    return 0;
}
