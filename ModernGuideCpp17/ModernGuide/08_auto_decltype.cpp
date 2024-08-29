/**
 * @file 08_auto_decltype.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** C++11中增加了很多新的特性,比如可以使用auto自动推导变量的类型
 * 还能够结合decltype来表示函数的返回值, 使用新的特性可以写出更加简洁,更加现代的代码.
 * 1. 在C++11之前auto和static是对应的, 表示变量是自动存储的,
 *  但是非static的局部变量默认都是自动存储的,因此这个关键字变得非常鸡肋,
 * 在C++11中赋予了新的含义, 使用这个关键字能够像别的语言一样自动推导出变量的实际类型.
 * 
 * 2. C++11中auto并不代表一种实际的数据类型,只是一个类型声明的 "占位符",
 *  auto并不是万能的在任意场景下都能够推导出变量的实际类型,
 *  使用auto声明的变量必须要进行初始化, 以让编译器推导出它的实际类型
 *  在编译时将auto占位符替换为真正的类型.
 * 
 * 3. auto的限制
 * --不能作为函数参数使用,因为只有在函数调用的时候才会给函数参数传递实参,auto要求必须要给修饰的变量赋值
 * --不能用于类的非静态成员变量的初始化
 * --不能使用auto关键字定义数组
 * --无法使用auto推导出模板参数
 *  
 * 4. auto的应用
 * --用于STL的容器遍历 
 * --用于泛型编程 
 *  
 * 5. 在某些情况下, 不需要或者不能定义变量,但是希望得到某种类型,
 *  这时候就可以使用C++11提供的decltype关键字了,它的作用是在编译器编译的时候推导出一个表达式的类型.
 *  decltype 是"declare type"的缩写，意思是"声明类型".
 *  decltype的推导是在编译期完成的,它只是用于表达式类型的推导,并不会计算表达式的值.
 *  
 * 6. 推导规则 
 *  --表达式为普通变量或者普通表达式或者类表达式,在这种情况下,使用decltype推导出的类型和表达式的类型是一致的
 *  --表达式是函数调用，使用decltype推导出的类型和函数返回值一致
 * !函数 func() 返回的是一个纯右值（在表达式执行结束后不再存在的数据，也就是临时性的数据,
 * !对于纯右值而言，只有类类型可以携带const、volatile限定符，除此之外需要忽略掉这两个限定符，
 * !因此推导出的变量d的类型为 int 而不是 const int.
 * !表达式是一个左值，或者被括号( )包围，使用 decltype推导出的是表达式类型的引用（如果有const、volatile限定符不能忽略）
 * 
 * 7. decltype的应用多出现在泛型编程中,比如编写一个类模板,在里边添加遍历容器的函数
 * 在泛型编程中，可能需要通过参数的运算来得到返回值的类型(返回类型后置)
 * 
 */

#include <iostream>
#include <list>
#include <map>

// int func(auto a, auto b) // error
int func(int a, int b)
{
    std::cout << "a: " << a << ", b: " << b << std::endl;
    return a + b;
}

class Test
{
    // auto v1 = 0; // error

    // static auto v2 = 0; // error,类的静态非常量成员不允许在类内部直接初始化

    static const auto v3 = 10; // ok
};

void func()
{
    int array[] = {1, 2, 3, 4, 5}; // 定义数组

    auto t1 = array; // ok, t1被推导为 int* 类型
    // auto t2[] = array;           // error, auto无法定义数组
    // auto t3[] = {1, 2, 3, 4, 5}; // error, auto无法定义数组
}

template<typename T>
struct TestTemplate
{
};

int funcTest()

{
    TestTemplate<double> t;
    // TestTemplate<auto>   t1 = t; // error, 无法推导出模板类型
    return 0;
}

// ----------------------------------
class T1
{
public:
    static int get()
    {
        return 10;
    }
};

class T2
{
public:
    static std::string get()
    {
        return "hello, world";
    }
};

template<class A>
void func(void)
{
    auto val = A::get();
    std::cout << "val: " << val << std::endl;
}

template<class A, typename B> // 添加了模板参数 B
void func_(void)
{
    B val = A::get();
    std::cout << "val: " << val << std::endl;
}

class TestDecltype
{
public:
    std::string      text;
    static const int value = 110;
};

template<class T>
class Container
{
public:
    void func(T &c)
    {
        for (m_it = c.begin(); m_it != c.end(); ++m_it)
        {
            std::cout << *m_it << " ";
        }
        std::cout << std::endl;
    }

private:
    decltype(T().begin()) m_it; // 这里不能确定迭代器类型
};

// 在C++11中增加了返回类型后置语法，说明白一点就是将decltype和auto结合起来完成返回类型的推导
// 符号 -> 后边跟随的是函数返回值的类型, auto 会追踪 decltype() 推导出的类型
// auto func(参数1, 参数2, ...) -> decltype(参数表达式)
template<typename T, typename U>
auto add(T t, U u) -> decltype(t + u)
{
    return t + u;
}

int &test_post(int &i)
{
    return i;
}

double test_post(double &d)
{
    d = d + 100;
    return d;
}

template<typename T>
// 返回类型后置语法
auto myFunc(T &t) -> decltype(test_post(t))
{
    return test_post(t);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    auto x = 3.14; // x 是浮点型 double
    std::cout << "The type is: " << typeid(x).name() << " and value=" << x << std::endl;

    auto y = 520; // y 是整形 int
    std::cout << "The type is: " << typeid(y).name() << " and value=" << y << std::endl;

    auto z = 'a'; // z 是字符型 char
    std::cout << "The type is: " << typeid(z).name() << " and value=" << z << std::endl;
    // auto        nb;       // error，变量必须要初始化
    // auto double nbl; // 语法错误, 不能修改数据类型

    /* auto还可以和指针、引用结合起来使用也可以带上const、volatile限定符,
    * 在不同的场景下有对应的推导规则，规则内容如下：
    * 1. 当变量不是指针或者引用类型时，推导的结果中不会保留const、volatile关键字
    * 2. 当变量是指针或者引用类型时，推导的结果中会保留const、volatile关键字 */
    int   temp = 110;
    auto *a    = &temp;
    auto  b    = &temp;
    auto &c    = temp;
    auto  d    = temp;

    int         tmp = 250;
    const auto  a1  = tmp;
    auto        a2  = a1;
    const auto &a3  = tmp;
    auto       &a4  = a3;

    // ===============================
    std::map<int, std::string> person;
    for (auto it = person.begin(); it != person.end(); ++it)
    {
        // do something
    }

    func<T1>();
    func<T2>();

    func_<T1, int>();         // 手动指定返回值类型 -> int
    func_<T2, std::string>(); // 手动指定返回值类型 -> string

    std::cout << "\n===========decltype===================\n";
    // 看到decltype推导的表达式可简单可复杂,auto是做不到的,auto只能推导已初始化的变量类型
    int                    a_ = 10;
    decltype(a_)           b_ = 99;       // b -> int
    decltype(a_ + 3.14)    c_ = 52.13;    // c -> double
    decltype(a_ + b_ * c_) d_ = 520.1314; // d -> double

    decltype(TestDecltype::value) c__ = 0;
    TestDecltype                  t;
    decltype(t.text)              d__ = "hello, world";

    std::cout << "\n===========decltype===================\n";
    const std::list<int> lst{1, 2, 3, 4, 5, 6, 7, 8, 9};

    Container<const std::list<int>> obj;
    obj.func(lst);

    int    x_integer = 520;
    double y_integer = 13.14;
    // auto   z_num     = add<int, double>(x_integer, y_integer);
    auto   z_num = add(x_integer, y_integer); // 简化之后的写法
    std::cout << "z_num: " << z_num << std::endl;

    int    x_post = 520;
    double y_post = 13.14;
    auto   z_post = myFunc(x_post); // 简化之后的写法
    std::cout << "z_post: " << z_post << std::endl;
    auto z1_post = myFunc(y_post); // 简化之后的写法
    std::cout << "z1_post: " << z1_post << std::endl;

    return 0;
}
