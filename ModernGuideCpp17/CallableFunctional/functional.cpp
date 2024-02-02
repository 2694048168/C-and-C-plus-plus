/**
 * @file functional.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Modern C++ function and bind
 * @version 0.1
 * @date 2024-02-02
 * 
 * @copyright Copyright (c) 2024
 * 
 * ========= function概念：
 * std::function是一个函数包装器模板，
 * 在c++11中，将function纳入标准库中，该函数包装器模板能包装任何类型的可调用元素.
 * 一个std::function类型对象实例可以包装下列这几种可调用元素类型：
 * 函数、函数指针、类成员函数指针或任意类型的函数对象（例如定义了operator()操作并拥有函数闭包）
 * 基本格式：
 * function<return-type(type1,type2)> func
 * 作用：实现接口统一
 * 
 * ========= bind
 * std::bind是一个标准库函数，定义在functional头文件中。
 * 可以将bind函数看作一个通用的函数适配器，它接受一个可调用对象，生成新的可调用对象来适应原对象的参数列表。
 * template< class F, class... Args >
 * std::bind( F&& f, Args&&... args );
 * 
 */

#include <functional>
#include <iostream>
#include <map>
#include <string>

int add(int x, int y)
{
    return x + y;
}

int sub(int x, int y)
{
    return x - y;
}

void func(std::function<int(int, int)> f, int x, int y)
{
    std::cout << f(x, y) << std::endl;
}

void print(const std::string &s, const std::string &s1)
{
    std::cout << s << s1 << std::endl;
}

class MyMath
{
public:
    MyMath()
    {
        std::cout << "默认构造函数" << std::endl;
    }

    MyMath(const MyMath &math)
    {
        std::cout << "默认拷贝函数" << std::endl;
    }

    int add(int x, int y)
    {
        return x + y;
    };

    static int sub(int x, int y)
    {
        return x - y;
    };
};

// -----------------------------------
int main(int argc, const char **argv)
{
    std::function<int(int, int)> f1 = add;
    std::function<int(int, int)> f2 = sub;
    std::function<int(int, int)> f3 = [](int x, int y)
    {
        return x * y;
    };

    func(f1, 10, 20);
    func(f2, 10, 20);
    func(f3, 10, 20);
    //------------------------------

    std::map<std::string, std::function<int(int, int)>> m{
        {"+", f1},
        {"-", f2},
        {"*", f3}
    };

    std::cout << m["+"](10, 20) << std::endl;
    std::cout << m["-"](10, 20) << std::endl;
    std::cout << m["*"](10, 20) << std::endl;

    // ===========================================
    std::cout << "==========================\n";
    // 1. 绑定普通函数
    std::function<void(const std::string &)> func  = std::bind(print, std::placeholders::_1, "bcd");
    std::function<void(const std::string &)> func1 = std::bind(print, "bcd", std::placeholders::_1);
    func("def");
    func1("def");

    // 2. 绑定类成员函数
    typedef int             (MyMath::*FUNC)(int, int);
    MyMath                  math_obj;
    // auto func_obj = std::bind(&MyMath::add, math_obj, placeholders::_1, 3);
    std::function<int(int)> func_obj = std::bind(&MyMath::add, &math_obj, std::placeholders::_1, 3);
    std::cout << func_obj(5) << std::endl;

    // 3. 绑定类静态成员函数
    std::function<int(int)> func_ = std::bind(MyMath::sub, std::placeholders::_1, 30);
    std::cout << func_(50) << std::endl;

    /**
     * @brief 注意：
     * ● bind预先绑定的参数需要传具体的变量或值进去，对于预先绑定的参数，是pass-by-value的；
     * ● 对于不事先绑定的参数，需要传std::placeholders进去，从_1开始，依次递增。placeholder是pass-by-reference的；
     * 如果要传递引用，则要使用std::ref()
     * 
     */
    int  num         = 41;
    auto func_lambda = [](int &a)
    {
        a = a + 1;
        std::cout << a << std::endl;
    };

    // auto func_bind = std::bind(func_lambda, std::ref(num));
    auto func_bind = std::bind(func_lambda, num);
    func_bind();
    std::cout << num << std::endl;

    return 0;
}
