/**
 * @file 04_typeCheckingTemplates.cpp
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
 * @brief Type Checking in Templates
 *  模板是类型安全的, 在模板实例化过程中, 编译器会复制粘贴模板参数,
 * !如果生成的代码不正确, 编译器则不会生成任何实例.
 * 
 * ==== 这些错误信息是模板初始化失败所产生的晦涩错误信息的典型例子
 *
 * 虽然模板实例化保证了类型安全, 但类型检查发生在编译过程的后期.
 * 当编译器实例化模板时, 它将模板参数类型粘贴到模板中, 插入类型后, 编译器尝试编译结果,
 * 如果实例化失败, 编译器(不同编译器实现不一致)就会在模板实例化时留下"遗言".
 *
 * C++ 模板编程与鸭子类型(duck-typed)的语言有相似之处,
 * 鸭子类型语言(Python)将类型检查推迟到运行时进行,
 * *其基本理念, 如果对象看起来像鸭子, 而且叫声也像鸭子, 那么它就是鸭子类型.
 * 不幸的是, 这意味着在执行程序之前, 无法从根本上知道对象是否支持某个特定的操作.
 * 对于模板, 无法知道实例化是否会成功, 直到尝试编译它.
 * 虽然鸭子类型语言可能在运行时崩溃, 但模板会在编译时崩溃.
 * C++ 界的意见领袖们普遍认为这种情况是不能接受的,所以诞生了一个精彩的解决方案: concept
 * 
 */

/**
 * @brief T 有一个隐含的要求: 它必须支持乘法运算.
 * 如果试图对 char* 使用平方函数 square, 编译会失败.
 * 
 */
template<typename T>
T square(T value)
{
    return value * value;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    int val = 12;
    printf("the square of %d is %d", val, square(val));

    // char my_char{'Q'};
    // !指针不支持乘法运算, 所以模板初始化失败
    // auto result = square(&my_char); // !ERROR Bang!

    return 0;
}
