/**
 * @file 00_operators.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <typeinfo>


/**
 * @brief 表达式是产生结果和副作用的计算过程,
 * 通常表达式包含进行运算的操作数和运算符, 许多运算符已融合到核心语言中.
 * - 内置运算符
 * - 重载运算符 new 
 * - 用户自定义字面量
 * - 探讨类型转换
 * - constexpr 常量表达式
 * - 被广泛误解的 volatile 关键字
 * 
 * ==== 运算符[如加法运算符（+）和地址运算符（&）]对
 * 被称为操作数的参数（如数值或对象）进行处理.
 * *逻辑、算术、赋值、自增/自减、比较、成员访问、三元条件和逗号运算符
 * 
 */

/**
  * @brief Operators
  * &&, ||, !, &, |, ^, ~, <<, >> 
  * 运算符逻辑与, 逻辑或, 逻辑非,
  * 按位逻辑运算符与, 或, 异或, 取补, 左移, 右移
  * 
  * ==== 算术运算符 Binary Arithmetic Operators Unary Arithmetic Operators
  * 一元和二元算术运算符可处理整数类型和浮点类型(也称为算术类型)
  * *一元加号运算符和一元减号运算符使用单个算术操作数
  * !这两个运算符都将其操作数提升为 int 类型.
  * !二元运算符会对其操作数进行整数提升.
  * 
  * 不要混淆类型转换和类型提升;
  * 类型转换是将一种类型的对象转换为另一种类型;
  * 类型提升是为解释字面量而设置的一组规则.
  * 
  * ==== 赋值运算符 Assignment Operators
  * ==== 自增和自减运算符 Increment and Decrement Operators
  * ==== 比较运算符 Comparison Operators
  * ==== 成员访问运算符 Member Access Operators
  * 1. 下标运算符 []
  * 2. 间接寻址运算符 *
  * 3. 地址运算符 &
  * 4. 对象成员运算符 .
  * 5. 成员指针运算符 -＞
  * ==== 三元条件运算符 Ternary Conditional Operator
  * ==== 逗号运算符 The Comma Operator
  * 
  */

/**
* @brief 三元条件运算符（即 x ? y ：z）是一个语法糖,
* 要三个操作数(三元), 将第一个操作数 x 作为布尔表达式, 
* 并根据表达式的值(true or false)来评估是返回第二个操作数 y 还是返回第三个操作数 z
* 
*/
int step(int input)
{
    return input > 0 ? 1 : 0;
}

//* ==== 逗号运算符, 各表达式从左到右依次求值，最右边的表达式的值是返回值
int confusing(int &x)
{
    return x = 9, x++, x / 2;
}

/**
 * @brief 重载运算符 Operator Overloading
 * 对于用户自定义类型, 可通过运算符重载方式为这些运算符指定自定义行为.
 * 只需用关键字 operator 和紧随其后的重载运算符,
 * 即可为用户自定义类指定运算符行为.
 * !注意, 请确保返回类型和参数与要处理的操作数类型匹配.
 * 
 */
struct CheckedInteger
{
    // 由于 m_value 参数是 const类型的
    // 因此 CheckedInteger 是不可修改的,  即在构造后无法修改 CheckedInteger的状态
    CheckedInteger(unsigned int value)
        : m_value{value}
    {
    }

    // overload operator+, 允许普通的 unsigned int 与CheckedInteger 相加
    // 以生成具有正确 m_value 的新的 CheckedInteger
    CheckedInteger operator+(unsigned int other) const
    {
        CheckedInteger result{m_value + other};

        // 每当加法导致 unsigned int 溢出时, 结果都将小于原始值
        // 检测该条件, 如果检测到溢出, 则抛出异常.
        if (result.m_value < m_value)
            throw std::runtime_error{"Overflow!"};

        return result;
    }

    const unsigned int m_value;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // *赋值运算符实际上并未采用提升规则，分配给操作数的类型并不会改变
    int x = 5;
    printf("the before type of x: %s\n", typeid(x).name());
    x /= 2.0f; // x is still int
    printf("the after type of x: %s\n", typeid(x).name());

    /**
     * @brief 运算符返回的值取决于运算符是前置还是后置,
     * 前置运算符将返回修改后的操作数的值,
     * 而后置运算符则返回修改前操作数的值
     * prefix or postfix
     */
    int val = 1;
    printf("the prefix: %d\n", ++val);      // output 2
    printf("and the postfix: %d\n", val++); // output 2

    // 比较运算符的操作数也将进行类型转换(提升),比较运算符也可应用于指针

    int  x_{};
    auto y = confusing(x_);
    printf("x: %d, and y: %d\n", x_, y);

    /**
     * @brief type_traits, 
     * https://en.cppreference.com/w/cpp/header/type_traits
     * 它允许在编译时进行类型特征提取, 
     * ＜limits＞头文件提供了与其相关的类型, 它允许查询算术类型的各种属性.
     * 在＜limits＞中, 模板类 numeric_limits 公开了许多成员常量,
     * 这些成员常量提供有关模板参数的信息, 例如 max() 方法返回了给定类型的最大有限值.
     * 
     */
    CheckedInteger a{100};
    auto           b = a + 200;
    printf("a + 200 = %u\n", b.m_value);

    try
    {
        auto c = a + std::numeric_limits<unsigned int>::max();
        printf("the value of c: %u\n", c.m_value);
    }
    catch (const std::overflow_error &e)
    {
        printf("(a + max) Exception: %s\n", e.what());
    }

    return 0;
}
