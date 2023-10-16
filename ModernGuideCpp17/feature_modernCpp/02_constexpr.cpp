/**
 * @file 02_constexpr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <iostream>

// C++11之前只有const关键字，从功能上来说这个关键字有双重语义：变量只读，修饰常量
// 变量只读并不等价于常量
void func(const int num)
{
    const int count = 24;

    // std::array<int, num>   array; /* error，num是一个只读变量，不是常量 */
    std::array<int, count> arr; /* ok，count是一个常量 */

    int a1 = 520;
    int a2 = 250;

    const int &b = a1;
    // b             = a2; /* error */
    a1 = 1314;
    std::cout << "b: " << b << std::endl; /* 输出结果为1314 */
}

// C++11中添加了一个新的关键字constexpr，这个关键字是用来修饰常量表达式的
// 常量表达式和非常量表达式的计算时机不同，非常量表达式只能在程序运行阶段计算出结果，
// 但是常量表达式的计算往往发生在程序的编译阶段，这可以极大提高程序的执行效率
// -------------------------------------------
// 在使用中建议将 const 和 constexpr 的功能区分开，
// 即凡是表达“只读”语义的场景都使用 const，表达“常量”语义的场景都使用 constexpr。
// -------------------------------------------
// 常量表达式函数 constexpr 可以修饰的函数：
// 普通函数/类成员函数、类的构造函数、模板函数
struct Person
{
    const char *name;
    int         age;
};

// 定义函数模板
template<typename T>
constexpr T display(T t)
{
    return t;
}

// ------------------------------
int main(int argc, char **argv)
{
    func(42);

    // ------------
    struct Person p
    {
        "luff", 19
    };
    //普通函数
    struct Person ret = display(p);
    std::cout << "luff's name: " << ret.name << ", age: " << ret.age << std::endl;

    //常量表达式函数
    constexpr int ret1 = display(250);
    std::cout << ret1 << std::endl;

    constexpr struct Person p1
    {
        "luff", 19
    };
    constexpr struct Person p2 = display(p1);
    std::cout << "luff's name: " << p2.name << ", age: " << p2.age << std::endl;

    return 0;
}