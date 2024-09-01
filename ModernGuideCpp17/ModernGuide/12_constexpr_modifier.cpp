/**
 * @file 12_constexpr_modifier.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * 在C++11之前只有const关键字,从功能上来说这个关键字有双重语义：变量只读，修饰常量.
 * !变量只读并不等价于常量
 *  
 * 在C++11中添加了一个新的关键字constexpr, 这个关键字是用来修饰常量表达式的.
 * 所谓常量表达式, 指的就是由多个（≥1）常量(值不会改变)组成并且在编译过程中就得到计算结果的表达式.
 * 在介绍gcc/g++工作流程, C++ 程序从编写完毕到执行分为四个阶段: 预处理、 编译、汇编和链接4个阶段,
 *  得到可执行程序之后就可以运行了.
 * *需要额外强调的是, 常量表达式和非常量表达式的计算时机不同,
 * *非常量表达式只能在程序运行阶段计算出结果, 但是常量表达式的计算往往发生在程序的编译阶段,
 * *这可以极大提高程序的执行效率,因为表达式只需要在编译阶段计算一次,节省了每次程序运行时都需要计算一次的时间.
 * 
 * C++11中添加constexpr 可以在程序中使用它来修饰常量表达式, 用来提高程序的执行效率.
 * ?在使用中建议将 const 和 constexpr 的功能区分开,
 * ?即凡是表达"只读"语义的场景都使用 const; 表达"常量"语义的场景都使用 constexpr;
 * 
 * 对于 C++ 内置类型的数据,可以直接用 constexpr 修饰;
 * 但如果是自定义的数据类型(用 struct 或者 class 实现),直接用 constexpr 修饰是不行的.
 * 
 * 2. 常量表达式函数
 * 为了提高C++程序的执行效率, 可以将程序中值不需要发生变化的变量定义为常量;
 * 也可以使用constexpr修饰函数的返回值, 这种函数被称作常量表达式函数,
 * 这些函数主要包括以下几种: 普通函数/类成员函数、类的构造函数、模板函数.
 * ---修饰函数(since c++11 not support, but c++17 support)
 *  *函数必须要有返回值, 并且return 返回的表达式必须是常量表达式
 *  *函数在使用之前, 必须有对应的定义语句
 *  *整个函数的函数体中,不能出现非常量表达式之外的语句(using 指令、typedef 语句以及 static_assert 断言、return语句除外)
 * ---修饰模板函数
 *  *C++11 语法中constexpr 可以修饰函数模板, 但由于模板中类型的不确定性,
 *  *因此函数模板实例化后的模板函数是否符合常量表达式函数的要求也是不确定的.
 *  *如果 constexpr 修饰的模板函数实例化结果不满足常量表达式函数的要求, 
 *  *则 constexpr 会被自动忽略,即该函数就等同于一个普通函数.
 * ---修饰构造函数
 *  *如果想用直接得到一个常量对象,也可以使用constexpr修饰一个构造函数,
 *  *这样就可以得到一个常量构造函数了. 常量构造函数有一个要求: 
 *  ?构造函数的函数体必须为空,并且必须采用初始化列表的方式为各个成员赋值.
 * 
 */

#include <iostream>

void func(const int num)
{
    const int count = 24;
    // int       array[num];    // !error，num是一个只读变量，不是常量
    int       array1[count]; // ok，count是一个常量

    int a1 = 520;
    int a2 = 250;

    // b是一个常量的引用，所以b引用的变量是不能被修改的
    const int &b = a1;
    // b             = a2; // !error
    std::cout << "b: " << b << std::endl;

    // 在const对于变量a1是没有任何约束的，a1的值变了b的值也就变了
    a1 = 1314;

    // 引用b是只读的，但是并不能保证它的值是不可改变的，也就是说它不是常量
    std::cout << "b: " << b << std::endl;
}

struct Test
{
    int id;
    int num;
};

constexpr int func_test()
{
    auto num = 42;
    return num;
}

constexpr int func1();

/*
// error
constexpr int func2()
{
    // constexpr int a = 100;
    // const int a = 100;
    int a = 100;
    // constexpr int b = 10;
    // const int b = 10;
    int b = 10;

    // 函数体内部的for循环是一个非法操作
    for (int i = 0; i < b; ++i)
    {
        std::cout << "i: " << i << std::endl;
    }
    return a + b;
}
*/

// ok
constexpr int func3()
{
    using my_type       = int;
    constexpr my_type a = 100;
    constexpr my_type b = 10;
    constexpr my_type c = a * b;
    return c - (a + b);
}

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

struct Person_test
{
    // 构造函数的函数体必须为空，并且必须采用初始化列表的方式为各个成员赋值
    // QT 中常见的方式(UI对应的类) 构造函数
    constexpr Person_test(const char *p, int age)
        : name(p)
        , age(age)
    {
    }

    const char *name;
    int         age;
};

// -------------------------------------
int main(int argc, const char **argv)
{
    func(42);

    const int     i  = 520;   // 是一个常量表达式
    const int     j  = i + 1; // 是一个常量表达式
    constexpr int i_ = 520;   // 是一个常量表达式
    constexpr int j_ = i + 1; // 是一个常量表达式
    std::cout << i << " " << j << " " << i_ << " " << j_ << std::endl;

    constexpr Test t{1, 2};
    constexpr int  id  = t.id;
    constexpr int  num = t.num;
    // t.num += 100; // error，不能修改常量
    std::cout << "id: " << id << ", num: " << num << std::endl;

    std::cout << "constexpr int func_test() return: " << func_test() << std::endl;
    std::cout << "constexpr int func1() return: " << func1() << std::endl;

    struct Person p
    {
        "lu_ffy", 19
    };
    //普通函数
    struct Person ret = display(p);
    std::cout << "lu_ffy's name: " << ret.name << ", age: " << ret.age << std::endl;

    //常量表达式函数
    constexpr int ret1 = display(250);
    std::cout << ret1 << std::endl;

    constexpr struct Person p1
    {
        "lu_ffy", 19
    };
    constexpr struct Person p2 = display(p1);
    std::cout << "lu_ffy's name: " << p2.name << ", age: " << p2.age << std::endl;

    constexpr struct Person_test p1_test("Ithaca", 29);
    std::cout << "Ithaca's name: " << p1.name << ", age: " << p1.age << std::endl;

    return 0;
}

constexpr int func1()
{
    constexpr int a = 100;
    return a;
}
