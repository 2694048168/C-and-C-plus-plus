/**
 * @file 08_moveSemantics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <utility>

/**
 * @brief Move Semantics 移动语义
 * 当涉及大量数据时, 复制语义在运行时可能相当耗时,
 * 通常情况下, 只是想把资源的所有权从一个对象转移到另一个对象,
 *  可以生成一个副本对象并销毁原始对象, 但这通常效率很低,
 * 相反可以使用移动语义(move semantics),移动语义是复制语义的必然结果,
 * 它要求对象y被移动到对象x中后, x相当于y之前的值.
 * 移动后, y处于一种特殊的状态, 称为"移出"(moved-from)状态,
 *  只能对"移出"对象进行两种操作: (重新)赋值或销毁.
 * NOTE: 注意将对象y移动到对象x中不只是重命名过程:它们都是单独的对象, 有单独的存储空间和单独的生命周期.
 * 与指定复制行为的方式类似, 可以通过移动构造函数和移动赋值运算符指定对象的移动.
 * 
 * 1. 复制行为可能浪费资源
 * 2. 移动行为是很危险的, 如果不小心使用了处于移出状态的对象,可能会招致灾难
 * 3. 编译器有内置的保障措施: 左值(lvalue)和右值(rvalue)
 * 
 * ===== Value Categories 值类别
 * 每个表达式都有两个重要的特征: 它的类型和值类别,
 * 值类别描述了什么样的操作对该表达式有效.
 * 由于C++的进化, 值类别变得很复杂: 表达式可以是广义左值(glvalue), 纯右值(prvalue)
 * 过期值(xvalue), 左值(不是过期值的广义左值)或者右值(纯右值或过期值).
 * 简单理解值类别: 左值是有名字的值, 而右值是非左值的值.
 * 
 * ==== lvalue and rvalue References 左值引用和右值引用
 * 可以使用左值引用和右值引用向编译器传达函数接受左值或右值作为参数的意思.
 * 这些参数用一个 & 表示 lvalue reference; 可以使用 && 来接受参数右值引用 rvalue reference.
 * 编译器在确定对象是左值还是右值方面做得很好,
 * 
 */
void ref_type(int &val)
{
    printf("lvalue reference %d\n", val);
};

// function overload
void ref_type(int &&val)
{
    printf("rvalue reference %d\n", val);
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // *词源是右值和左值, 左和右指的是在构造过程中各自相对于等号的位置.
    // *NOTE: ISO C++标准在[basic]和[expr]节详细解释了值类别
    // https://en.cppreference.com/w/cpp/language/value_category
    int val{24};  // 24 is rvalue
    int num{val}; // val is lvalue
    printf("left value: %d, right value: %d\n", val, num);

    // ============= lvalue and rvalue reference =============
    auto x = 12;
    ref_type(x);     // call lvalue reference
    ref_type(2);     // call rvalue reference
    ref_type(x + 2); // call rvalue reference

    /**
     * @brief std::move函数
     * 使用 <utility>头文件中的 std::move 函数将左值引用转换成右值引用
     * NOTE: C++委员会也许应该把 std::move 命名为 std::rvalue,
     * std::move 函数实际上并不移动任何东西, 它只是转换类型.
     * !当使用 std::move 时, 要非常小心, 因为它删除了阻止与处于移出状态的对象进行交互的保障措施.
     * 
     */
    printf("========= std::move ========\n");
    ref_type(std::move(x));
    ref_type(42);
    ref_type(42 - 1);

    /**
     * @brief Move Construction 移动构造
     * 移动构造函数看起来像复制构造函数, 只是它们采用右值引用而非左值引用.
     * 执行这个移动构造函数比执行复制构造函数的成本低很多.
     * 移动构造函数被设计成不会抛出异常, 可以把它标记为 noexcept,
     * 首选应该是使用 noexcept 移动构造函数, 编译器不能使用抛出异常的移动构造函数, 
     * 而会使用复制构造函数, 因为编译器更喜欢慢但正确的代码, 而不是快但不正确的代码.
     * 
     */

    /**
      * @brief Move Assignment 移动赋值
      * 可以通过 operator= 创建类似于复制赋值运算符的移动赋值运算符,
      * 移动赋值运算符接受右值引用, 而不是 const 型左值引用, 通常将它标记为 noexcept.
      * 可以使用右值引用语法和 noexcept 限定符来声明移动赋值运算符, 就像移动构造函数一样自引用检查.
      * 
      * 如果这里不需要 std::move 的话?这可能会导致难以诊断的bug.
      * !请记住, 不能使用处于移出状态的对象, 除非释放或销毁它们, 任何其他操作都将产生未定义行为.
      */

    return 0;
}
