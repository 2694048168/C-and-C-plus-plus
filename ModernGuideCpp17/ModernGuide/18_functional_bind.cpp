/**
 * @file 18_functional_bind.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 可调用对象
 * 在C++中存在"可调用对象"这么一个概念:
 * *---是一个函数指针;
 * *---具有operator()成员函数的类对象（仿函数）;
 * *---可被转换为函数指针的类对象;
 * *---类成员函数指针或者类成员指针;
 * C++中的可调用类型虽然具有比较统一的操作形式,但定义方式五花八门,
 * 这样在试图使用统一的方式保存,或者传递一个可调用对象时会十分繁琐.
 * ?现代C++11通过提供std::function 和 std::bind统一了可调用对象的各种操作.
 * 
 * 2. 可调用对象包装器
 * std::function是可调用对象的包装器. 
 * 它是一个类模板,可以容纳除了类(非静态)成员（函数）指针之外的所有可调用对象.
 * 通过指定它的模板参数,它可以用统一的方式处理函数、函数对象、函数指针，并允许保存和延迟执行它们.
 * #include <functional>
 * *std::function<返回值类型(参数类型列表)> diy_name = 可调用对象;
 * ---统一的方式处理
 * ---作为回调函数使用(回调函数本身就是通过函数指针实现的,使用对象包装器可以取代函数指针的作用)
 * 
 * 3. 绑定器
 * *std::bind用来将可调用对象与其参数一起进行绑定.
 * ?绑定后的结果可以使用std::function进行保存,并延迟调用到任何需要的时候.
 * ---将可调用对象与其参数一起绑定成一个仿函数
 * ---将多元(参数个数为n，n>1)可调用对象转换为一元或者(n-1)元可调用对象,即只绑定部分参数
 * // 绑定非类成员函数/变量
 * *auto f = std::bind(可调用对象地址, 绑定的参数/占位符);
 * // 绑定类成员函/变量
 * *auto f = std::bind(类函数/成员地址, 类实例对象地址, 绑定的参数/占位符);
 * 
 * ?可调用对象包装器std::function是不能实现对类成员函数指针或者类成员指针的包装的,
 * ?但是通过绑定器std::bind的配合之后,就可以完美的解决这个问题了.
 * 
 */

#include <functional>
#include <iostream>
#include <vector>

// ==============统一的方式处理
int add(int a, int b)
{
    std::cout << a << " + " << b << " = " << a + b << std::endl;
    return a + b;
}

class T1
{
public:
    static int sub(int a, int b)
    {
        std::cout << a << " - " << b << " = " << a - b << std::endl;
        return a - b;
    }
};

class T2
{
public:
    int operator()(int a, int b)
    {
        std::cout << a << " * " << b << " = " << a * b << std::endl;
        return a * b;
    }
};

// ==============作为回调函数使用
class A
{
public:
    // 构造函数参数是一个包装器对象
    // 使用std::function作为函数的传入参数,可以将定义方式不相同的可调用对象进行统一的传递
    A(const std::function<void()> &f)
        : callback(f)
    {
    }

    void notify()
    {
        callback(); // 调用通过构造函数得到的函数指针
    }

private:
    std::function<void()> callback;
};

class B
{
public:
    void operator()()
    {
        std::cout << "我是要成为海贼王的男人!!!\n";
    }
};

// ==============绑定器
void callFunc(int x, const std::function<void(int)> &func)
{
    if (x % 2 == 0)
    {
        func(x);
    }
}

void output(int x)
{
    std::cout << x << " ";
}

void output_add(int x)
{
    std::cout << x + 10 << " ";
}

void output_perf(int x, int y)
{
    std::cout << x << " and " << y << std::endl;
}

// 可调用对象包装器std::function是不能实现对类成员函数指针或者类成员指针的包装的
// 但是通过绑定器std::bind的配合之后，就可以完美的解决这个问题了
class Test
{
public:
    void output(int x, int y)
    {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }

    int m_number = 100;
};

// --------------------------------------
int main(int argc, const char **argv)
{
    // std::function可以将可调用对象进行包装,得到一个统一的格式,
    // 包装完成得到的对象相当于一个函数指针,和函数指针的使用方式相同,
    // 通过包装器对象就可以完成对包装的函数的调用了.
    std::vector<std::function<int(int, int)>> func_vec;

    // 绑定一个普通函数
    func_vec.emplace_back(add);
    // 绑定一个静态类成员函数
    func_vec.emplace_back(T1::sub);
    // 绑定一个仿函数
    T2 t;
    func_vec.emplace_back(t);

    // 函数调用
    for (const auto &func : func_vec)
    {
        func(9, 3);
    }

    // ==============作为回调函数使用
    // 使用对象包装器std::function可以非常方便的将仿函数转换为一个函数指针,
    // 通过进行函数指针的传递,在其他函数的合适的位置就可以调用这个包装好的仿函数了
    B b;
    A a(b); // 仿函数通过包装器对象进行包装
    a.notify();

    // ==============绑定器
    std::cout << "==============\n";
    // 使用绑定器绑定可调用对象和参数
    auto f1 = std::bind(output, std::placeholders::_1);
    for (int i = 0; i < 10; ++i)
    {
        callFunc(i, f1);
    }
    std::cout << std::endl;
    /* 使用了std::bind绑定器,在函数外部通过绑定不同的函数,控制了最后执行的结果.
    std::bind绑定器返回的是一个仿函数类型,得到的返回值可以直接赋值给一个std::function,
    在使用的时候我们并不需要关心绑定器的返回值类型,使用auto进行自动类型推导就可以了.
    placeholders::_1是一个占位符,代表这个位置将在函数调用时被传入的第一个参数所替代;
    其他的占位符placeholders::_2、placeholders::_3、placeholders::_4、placeholders::_5等
    有了占位符的概念之后，使得std::bind的使用变得非常灵活.  */
    auto f2 = std::bind(output_add, std::placeholders::_1);
    for (int i = 0; i < 10; ++i)
    {
        callFunc(i, f2);
    }
    std::cout << std::endl;

    // 使用绑定器绑定可调用对象和参数, 并调用得到的仿函数
    std::bind(output_perf, 1, 2)();
    std::bind(output_perf, std::placeholders::_1, 2)(10);
    std::bind(output_perf, 2, std::placeholders::_1)(10);
    // *std::bind可以直接绑定函数的所有参数,也可以仅绑定部分参数.
    // *在绑定部分参数的时候,通过使用std::placeholders来决定空位参数将会属于调用发生时的第几个参数.
    // error, 调用时没有第二个参数
    // std::bind(output_perf, 2, std::placeholders::_2)(10);
    // 调用时第一个参数10被吞掉了，没有被使用
    std::bind(output_perf, 2, std::placeholders::_2)(10, 20);
    std::bind(output_perf, std::placeholders::_1, std::placeholders::_2)(10, 20);
    std::bind(output_perf, std::placeholders::_2, std::placeholders::_1)(10, 20);

    // 可调用对象包装器std::function是不能实现对类成员函数指针或者类成员指针的包装的
    // 但是通过绑定器std::bind的配合之后，就可以完美的解决这个问题了
    std::cout << "==============\n";
    Test                          t__;
    // 绑定类成员函数
    std::function<void(int, int)> f1__ = std::bind(&Test::output, &t__, std::placeholders::_1, std::placeholders::_2);
    // 绑定类成员变量(公共)
    std::function<int &(void)>    f2__ = std::bind(&Test::m_number, &t__);

    // 调用
    f1__(520, 1314);
    f2__() = 2333;
    std::cout << "t.m_number: " << t__.m_number << std::endl;

    /* 在用绑定器绑定类成员函数或者成员变量的时候需要将它们所属的实例对象一并传递到绑定器函数内部.
    f1__的类型是function<void(int, int)>,
    通过使用std::bind将Test的成员函数output的地址和对象t绑定,并转化为一个仿函数并存储到对象f1中.
    使用绑定器绑定的类成员变量m_number得到的仿函数被存储到了类型为function<int&(void)>的包装器对象f2__中,
    并且可以在需要的时候修改这个成员.
    其中int是绑定的类成员的类型,并且允许修改绑定的变量,因此需要指定为变量的引用,由于没有参数因此参数列表指定为void. */

    return 0;
}
