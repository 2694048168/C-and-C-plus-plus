/**
 * @file 00_functionDeclarations.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 函数将代码封装成可重用的组件,
 * 介绍修饰符, 说明符, 返回类型出现在函数声明中, 并专门说明函数的行为.
 * 1. 重载解析和接受可变数量的参数
 * 2. 然后探索函数指针
 * 3. 类型别名
 * 4. 函数对象
 * 5. lambda 表达式
 * 6. std::function
 * 7. main 函数和命令行参数
 * 
 * ==== Function declarations
 * prefix-modifiers return-type func-name(arguments) suffix-modifiers;
 * 可以为函数提供许多可选的修饰符(或说明符),修饰符会以某种方式改变函数的行为;
 * 有些修饰符出现在函数声明或定义的开头(前缀修饰符), 前缀修饰符出现在返回类型之前;
 * 有些修饰符则出现在函数的结尾(后缀修饰符), 后缀修饰符出现在参数列表之后.
 * 
 * ---- Prefix Modifiers
 * 1. static 表示不是类成员的函数(自由函数)具有内部链接, 意味着该函数不会在该编译单元外部使用;
 *    static 关键字具有双重职责: 如果修饰一个方法(即类内部的函数), 
 *           则表明该函数与该类的实例无关, 而与该类本身相关联.
 * 2. virtual 表示方法可以被子类覆写, override 修饰符告知编译器子类打算覆写父类的虚函数.
 * 3. constexpr 指示应尽可能在编译时执行函数.
 * 4. [[noreturn]] 表示此函数无返回值(属性), 此属性可帮助编译器优化代码.
 * 5. inline 在优化代码时它起着指导编译器的作用.
 * 
 * *inline ---> 函数调用会被编译为一系列指令
 * 1）将参数放入寄存器和调用栈上;
 * 2）将返回地址推入调用栈中;
 * 3）跳转至被调用的函数;
 * 4）在函数完成后，跳转至返回地址;
 * 5）清理调用栈.
 * ?这些步骤通常执行得非常快, 如果在许多地方都会使用某个函数, 那么减少二进制文件大小的好处是巨大的.
 * 内联函数意味着将函数的内容直接复制并粘贴到执行路径中, 从而无须执行上面列出的五个步骤,
 * 这意味着当处理器执行代码时, 它将立即执行该函数的代码, 而无须执行函数调用所需的(适度)仪式性步骤,
 * 如果希望这种速度的边际增加优于二进制文件大小增加所带来的相应成本, 则可以使用 inline 关键字告知编译器,
 * inline 关键字提示编译器的优化器将函数直接内联而不是执行函数调用.
 * 在函数中添加 inline 不会改变其行为, 纯粹是对编译器的偏好表达,
 * 必须确保如果定义函数为 inline, 则必须在所有编译单元(translation unit)中都进行 inline 定义
 * NOTE: 现代编译器通常会在有意义的地方内联函数(尤其是如果未在单个编译单元之外使用某个函数时).
 * 
 * ---- Suffix Modifiers
 * 1. noexcept 表示函数永远不会抛出异常, 可以启用某些优化
 * 2. const 表示方法不会修改其类的实例, 
 *    *同时也表明允许 const 引用类型调用该方法
 * 3. final 修饰符表示子类不能覆写方法, 实际上这与 virtual 功能相反
 *    *将 final 关键字应用于整个类, 从而禁止该类完全成为父类
 * 4. volatile 对象的值可以随时更改, 因此出于优化目的, 编译器必须将对 volatile 对象的
 *    所有访问都视为可见的副作用. volatile 关键字表示可以在 volatile 对象上调用方法;
 *    这类似于 const 方法可在 const 对象上调用, 这两个关键字共同定义了方法的 const/volatile 限定,
 *    *cv 限定[const/volatile qualification](https://en.cppreference.com/w/cpp/language/cv)
 * 
 */

/**
 * @brief 每当使用接口继承时, 都应将实现类标记为 final,
 * 因为该修饰符可以鼓励编译器执行称为"去虚化"(de-virtualization)优化,
 * 当对虚调用进行去虚后, 编译器消除了与虚调用关联的运行时开销.
 * 
 */
struct BostonCor
{
    virtual void shoot() final
    {
        printf("What a God we have...God avenged Abraham Lincoln\n");
    }
};

struct BostonCorJunior : BostonCor
{
    // void shoot() override {} // !Bang shoot is final.
};

// const/volatile qualification
struct Distillate
{
    int apply() volatile
    {
        return ++applications;
    }

private:
    int applications{};
};

// ------------------------------------
int main(int argc, const char **argv)
{
    BostonCorJunior junior;
    junior.shoot();

    printf("==== The const/volatile qualification =====\n");
    volatile Distillate ethanol;
    printf("%d Tequila\n", ethanol.apply());
    printf("%d Tequila\n", ethanol.apply());
    printf("%d Tequila\n", ethanol.apply());
    printf("Floor!\n");

    return 0;
}
