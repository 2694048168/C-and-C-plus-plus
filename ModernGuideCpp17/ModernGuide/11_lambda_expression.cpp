/**
 * @file 11_lambda_expression.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 基本用法
lambda表达式是C++11最重要也是最常用的特性之一，这是现代编程语言的一个特点，lambda表达式有如下的一些优点：
1. 声明式的编程风格：就地匿名定义目标函数或函数对象，不需要额外写一个命名函数或函数对象。
2. 简洁：避免了代码膨胀和功能分散，让开发更加高效。
3. 在需要的时间和地点实现功能闭包，使程序更加灵活。
lambda表达式定义了一个匿名函数，并且可以捕获一定范围内的变量。lambda表达式的语法形式简单归纳如下：
[capture](params) opt -> ret {body;};
其中capture是捕获列表，params是参数列表，opt是函数选项，ret是返回值类型，body是函数体。
1. 捕获列表[]: 捕获一定范围内的变量
2. 参数列表(): 和普通函数的参数列表一样，如果没有参数参数列表可以省略不写
3. opt 选项， 不需要可以省略
    mutable: 可以修改按值传递进来的拷贝（注意是能修改拷贝，而不是值本身）
    exception: 指定函数抛出的异常，如抛出整数类型的异常，可以使用throw();
4. 返回值类型：在C++11中，lambda表达式的返回值是通过返回值后置语法来定义的
5. 函数体：函数的实现，这部分不能省略，但函数体可以为空
 * 
 * 2. 捕获列表
1. [] - 不捕捉任何变量
2. [&] - 捕获外部作用域中所有变量, 并作为引用在函数体内使用 (按引用捕获)
3. [=] - 捕获外部作用域中所有变量, 并作为副本在函数体内使用 (按值捕获)
        拷贝的副本在匿名函数体内部是只读的
4. [=, &foo] - 按值捕获外部作用域中所有变量, 并按照引用捕获外部变量 foo
5. [bar] - 按值捕获 bar 变量, 同时不捕获其他变量
6. [&bar] - 按引用捕获 bar 变量, 同时不捕获其他变量
7. [this] - 捕获当前类中的this指针
        让lambda表达式拥有和当前类成员函数同样的访问权限
        如果已经使用了 & 或者 =, 默认添加此选项
在匿名函数内部，需要通过lambda表达式的捕获列表控制如何捕获外部变量，以及访问哪些变量.
默认状态下lambda表达式无法修改通过复制方式捕获外部变量,如果希望修改这些外部变量,需要通过引用的方式进行捕获.
 * 
 * 3. 返回值
1. lambda表达式的返回值是非常明显的，因此在C++11中允许省略lambda表达式的返回值
 * 
 * 4. 函数本质
使用lambda表达式捕获列表捕获外部变量，如果希望去修改按值捕获的外部变量，那么应该如何处理呢？
这就需要使用mutable选项,被mutable修改是lambda表达式就算没有参数也要写明参数列表,
并且可以去掉按值捕获的外部变量的只读（const）属性.
 * lambda表达式在C++中会被看做是一个仿函数,
 * 因此可以使用std::function和std::bind来存储和操作lambda表达式 
 * 对于没有捕获任何变量的lambda表达式，还可以转换成一个普通的函数指针
 * 
 */

#include <functional>
#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    // 完整的lambda表达式定义
    auto func_full = [](int a) -> int
    {
        return a + 10;
    };
    std::cout << func_full(32) << std::endl;

    // 忽略返回值的lambda表达式定义
    // 一般情况下不指定lambda表达式的返回值,编译器会根据return语句自动推导返回值的类型,
    // 但需要注意的是 lambda 表达式不能通过列表初始化自动推导出返回值类型
    auto func = [](int a)
    {
        return a + 10;
    };
    std::cout << func(32) << std::endl;

    // ==========================
    int  a  = 0;
    auto f1 = [=]
    {
        // return a++; // error, 按值捕获外部变量, a是只读的
        return a;
    };

    /* 
    最后再剖析一下为什么通过值拷贝的方式捕获的外部变量是只读的:
    lambda表达式的类型在C++11中会被看做是一个带operator()的类,即仿函数.
    按照C++标准，lambda表达式的operator()默认是const的,一个const成员函数是无法修改成员变量值的.
    mutable选项的作用就在于取消operator()的const属性
     */

    auto f2 = [=]() mutable
    {
        return a++; // ok
    };

    // lambda表达式在C++中会被看做是一个仿函数，因此可以使用std::function和std::bind来存储和操作lambda表达式
    // 包装可调用函数
    std::function<int(int)> func_lambda = [](int a)
    {
        return a;
    };
    // 绑定可调用函数
    std::function<int(int)> func_callable = std::bind([](int a) { return a; }, std::placeholders::_1);

    // 函数调用
    std::cout << func_lambda(100) << std::endl;
    std::cout << func_callable(200) << std::endl;

    using func_ptr = int (*)(int);
    // 没有捕获任何外部变量的匿名函数, 转换成一个普通的函数指针
    func_ptr func_pure = [](int a)
    {
        return a;
    };
    // 函数调用
    std::cout << "the func_ptr return value: " << func_pure(1314);

    return 0;
}
