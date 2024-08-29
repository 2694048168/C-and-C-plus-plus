/**
 * @file 02_init_members.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/** 1. C++98 标准的类成员初始化
 * 在C++98中，支持了在类声明中使用等号 = 加初始值 的方式，来初始化类中静态成员常量,
 * 这种声明方式我们也称之为"就地"声明, 而非静态成员变量的初始化则必须在构造函数中进行.
 * 
 * 2. C++11 标准的类成员初始化
 * *初始化类的非静态成员, C++11标准对于C++98做了补充,
 * *允许在定义类的时候在类内部直接对非静态成员变量进行初始化,
 * *在初始化的时候可以使用等号 = 也可以使用花括号 {}(建议统一使用).
 * 
 * ?类内部赋值和初始化列表
 * 
 */

// C++98
struct Base
{
    Base()
        : a(250)
    {
    }

    Base(int num)
        : a(num)
    {
    }

    int                      a;
    int                      b = 1;
    // !类的静态成员，必须在类的外部进行初始化
    // static int               c = 0;
    static int               c;
    // static const double      d = 3.14;
    static const double      d;
    // !类的静态常量成员，但不是整形或者枚举，无法通过编译
    // static const char *const e = "i am Ithaca";
    static const char *const e;
    // *程序中的 static const 和 const static 是等价的
    static const int         f = 0;
};

int               Base::c = 110;
const double      Base::d = 3.14;
const char *const Base::e = "i am Ithaca";

// C++11 Modern Cpp
class Test
{
private:
    int         a = 9;
    int         b = {5};
    int         c{12};
    double      array[4] = {3.14, 3.15, 3.16, 3.17};
    double      array1[4]{3.14, 3.15, 3.16, 3.17};
    // std::string s1("hello"); // !error, 不能使用小括号() 初始化对象，应该使用花括号{}
    std::string s2{"hello, world"};
};

// 类内部赋值和初始化列表
class Init
{
public:
    // 使用初始化列表对类的非静态成员进行初始化
    Init(int x, int y, int z)
        : a(x)
        , b(y)
        , c(z)
    {
    }

    // 在类内部对非静态成员变量就地初始化（C++11新特性）
    int a = 1;
    int b = 2;
    int c = 3;
};

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "c = " << Base::c << '\n';
    std::cout << "d = " << Base::d << '\n';
    std::cout << "e = " << Base::e << '\n';
    std::cout << "f = " << Base::f << '\n';

    /* 在类内部就地初始化和初始化列表并不冲突,程序可以正常运行,
    程序员可以为同一成员变量既在类内部就地初始化, 又在初始化列表中进行初始化,
    只不过初始化列表总是看起来后作用于非静态成员.
    也就是说，通过初始化列表指定的值会覆盖就地初始化时指定的值. */
    Init tmp(10, 20, 30);
    std::cout << "\na: " << tmp.a << ", b: " << tmp.b << ", c: " << tmp.c << std::endl;

    return 0;
}
