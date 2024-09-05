/**
 * @file 19_POD_type.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. POD 类型
 * POD是英文中 Plain Old Data 的缩写,翻译过来就是普通的旧数据.
 * POD在C++中是非常重要的一个概念,通常用于说明一个类型的属性,尤其是用户自定义类型的属性.
 * POD属性在C++11中往往又是构建其他C++概念的基础,事实上在C++11标准中,POD出现的概率相当高.
 * *Plain: 表示是个普通的类型;
 * *Old: 体现了其与C的兼容性,支持标准C函数;
 * 在C++11中将 POD划分为两个基本概念的合集: 平凡的(trivial)和标准布局的(standard layout) 
 * 
 * 2. "平凡"类型
 * 一个平凡的类或者结构体应该符合以下几点要求:
 * *---拥有平凡的默认构造函数(trivial constructor)和析构函数(trivial destructor); 
 * ?编译器默认的/或者default显式声明的, 否则都是非平凡的
 * *---拥有平凡的拷贝构造函数(trivial copy constructor)和移动构造函数(trivial move constructor);
 * ?平凡的拷贝构造函数基本上等同于使用memcpy 进行类型的构造;
 * ?同平凡的默认构造函数一样，不声明拷贝构造函数的话，编译器会帮程序员自动地生成;
 * ?可以显式地使用=default 声明默认拷贝构造函数;
 * ?而平凡移动构造函数跟平凡的拷贝构造函数类似，只不过是用于移动语义;
 * *---拥有平凡的拷贝赋值运算符(trivial assignment operator)和移动赋值运算符(trivial move operator);
 * *---不包含虚函数以及虚基类
 * ?类中使用virtual 关键字修饰的函数 叫做虚函数;
 * ?虚基类是在创建子类的时候在继承的基类前加virtual 关键字 修饰;
 * 
 * 3. "标准布局"类型
 * 标准布局类型主要主要指的是类或者结构体的结构或者组合方式.
 * 标准布局类型的类应该符合以下五点定义,最重要的为前两条:
 * *---所有非静态成员有 相同 的访问权限(public，private，protected); 
 * *---在类或者结构体继承时，满足以下两种情况之一∶
 * ?派生类中有非静态成员，基类中包含静态成员(或基类没有变量);
 * ?基类有非静态成员，而派生类没有非静态成员;
 * ?非静态成员只要同时出现在派生类和基类间，即不属于标准布局;
 * ?对于多重继承，一旦非静态成员出现在多个基类中，即使派生类中没有非静态成员变量，派生类也不属于标准布局;
 * *---子类中第一个非静态成员的类型与其基类不同;
 * ?这样规定的目的主要是是节约内存，提高数据的读取效率.
 * *---没有虚函数和虚基类
 * *---所有非静态数据成员均符合标准布局类型，其基类也符合标准布局，这是一个递归的定义
 * 
 * 4. 对POD类型的判断
 * *---"平凡"类型判断: C++11提供的类模板叫做 is_trivial;
 * *---对"标准布局"类型的判断: 在C++11中可以使用模板类来帮助判断类型是否是一个标准布局的类型;
 * 
 */

#include <iostream>

class A
{
};

class B
{
    B() {}
};

class C : B
{
};

class D
{
    virtual void fn() {}
};

class E : virtual public A
{
};

// ----------------------------
struct A_
{
};

struct B_ : A_
{
    int j;
};

struct C_
{
public:
    int a;

private:
    int c;
};

struct D1_
{
    static int i;
};

struct D2_
{
    int i;
};

struct E1_
{
    static int i;
};

struct E2_
{
    int i;
};

struct D_
    : public D1_
    , public E1_
{
    int a;
};

struct E_
    : public D1_
    , public E2_
{
    int a;
};

struct F_
    : public D2_
    , public E2_
{
    static int a;
};

struct G_ : public A_
{
    int foo;
    A_  a;
};

struct H_ : public A_
{
    A_  a;
    int foo;
};

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << std::boolalpha;
    std::cout << "is_trivial:" << std::endl;
    std::cout << "int: " << std::is_trivial<int>::value << std::endl;
    std::cout << "A: " << std::is_trivial<A>::value << std::endl;
    std::cout << "B: " << std::is_trivial<B>::value << std::endl;
    std::cout << "C: " << std::is_trivial<C>::value << std::endl;
    std::cout << "D: " << std::is_trivial<D>::value << std::endl;
    std::cout << "E: " << std::is_trivial<E>::value << std::endl;

    std::cout << std::boolalpha << std::endl;
    std::cout << "is_standard_layout:" << std::endl;
    std::cout << "A: " << std::is_standard_layout<A_>::value << std::endl;
    std::cout << "B: " << std::is_standard_layout<B_>::value << std::endl;
    std::cout << "C: " << std::is_standard_layout<C_>::value << std::endl;
    std::cout << "D: " << std::is_standard_layout<D_>::value << std::endl;
    std::cout << "D1: " << std::is_standard_layout<D1_>::value << std::endl;
    std::cout << "E: " << std::is_standard_layout<E_>::value << std::endl;
    std::cout << "F: " << std::is_standard_layout<F_>::value << std::endl;
    std::cout << "G: " << std::is_standard_layout<G_>::value << std::endl;
    std::cout << "H: " << std::is_standard_layout<H_>::value << std::endl;

    // 事实上 C++使用的很多内置类型默认都是 POD的,
    // POD 最为复杂的地方还是在类或者结构体的判断.
    // 那么,使用POD有什么好处呢？
    // *1. 字节赋值, 代码中可以安全地使用memset 和 memcpy 对 POD类型进行初始化和拷贝等操作;
    // *2. 提供对C内存布局兼容, C++程序可以与C 函数进行相互操作, 因为POD类型的数据在C与C++ 间的操作总是安全的;
    // *3. 保证了静态初始化的安全有效,静态初始化在很多时候能够提高程序的性能,而POD类型的对象初始化往往更加简单;

    return 0;
}
