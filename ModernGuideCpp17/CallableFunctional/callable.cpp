/**
 * @file callable.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Modern C++ 中可调用对象
 * @version 0.1
 * @date 2024-02-02
 * 
 * @copyright Copyright (c) 2024
 * 
 * 一组执行任务的语句都可以视为一个"函数"，一个可调用对象
 * 在C++中就func的类型可以为：
 * ● 普通函数
 * ● 类成员函数
 * ● 类静态函数
 * ● 仿函数
 * ● 函数指针: 可被转换为函数指针的类对象 ---> operator type() 隐式类型转换
 * ● lambda表达式 C++11加入标准
 * 
 */

#include <iostream>

// ==================================
int ordinary_func_add(int x, int y);

typedef int           (*FUNC_ADD)(int, int);
using FUNC_ADD1 = int (*)(int, int);

// ==================================
class MyMath
{
public:
    int add(int x, int y)
    {
        return x + y;
    }

    static int sub(int x, int y)
    {
        return x - y;
    }

    int operator()(int x, int y)
    {
        return x * y;
    }
};

typedef int       (MyMath::*FUNC)(int, int);
using FUNC1 = int (MyMath::*)(int, int);

typedef int           (*FUNC_SUB)(int, int);
using FUNC_SUB1 = int (*)(int, int);

// =======================================================
// 可被转换为函数指针的类对象 ---> operator type() 隐式类型转换
int sum(const int &a, const int &b)
{
    return a + b;
}

// typedef int (*FUNC_SUM)(const int &a, const int &b);
using FUNC_SUM = int (*)(const int &a, const int &b);

class CallableMath
{
public:
    int a;
    CallableMath(int i)
        : a(i){};

    // operator type() 隐式类型转换
    operator int()
    {
        return a;
    }

    // operator type() 隐式类型转换
    operator FUNC_SUM()
    {
        return static_obj_sum;
    }

    int obj_sum(const int &a, const int &b)
    {
        return a + b;
    }

    static int static_obj_sum(const int &a, const int &b)
    {
        return a + b;
    }
};

// ------------------------------------
int main(int argc, const char **argv)
{
    // ==================================
    FUNC_ADD add_func = ordinary_func_add;
    std::cout << add_func(24, 42) << std::endl;

    FUNC_ADD1 add_func1 = ordinary_func_add;
    std::cout << add_func1(24, 42) << std::endl;
    std::cout << "==================================\n";

    MyMath my_math;
    FUNC1  func_math = &MyMath::add;
    std::cout << (my_math.*func_math)(10, 20) << std::endl;

    FUNC_SUB1 func_sub = MyMath::sub;
    std::cout << func_sub(24, 12) << std::endl;

    // 仿函数 functor
    std::cout << my_math(2, 12) << std::endl;
    FUNC func_mul = &MyMath::operator();
    std::cout << (my_math.*func_mul)(2, 1) << std::endl;
    std::cout << "==================================\n";

    // =======================================================
    // 可被转换为函数指针的类对象 ---> operator type() 隐式类型转换
    CallableMath   m(10);
    typedef int    (*FUNC_SUM)(const int &a, const int &b);
    typedef int    (CallableMath::*func)(const int &a, const int &b);
    using f1 = int (CallableMath::*)(const int &a, const int &b);

    int *a = new int(10);

    FUNC_SUM ff = CallableMath::static_obj_sum;
    int      x  = ff(4, 5);
    std::cout << x << std::endl;
    f1 f  = &CallableMath::obj_sum;
    x     = (m.*f)(3, 4);
    int y = m;
    std::cout << x << std::endl;
    FUNC_SUM ff1 = [](const int &a, const int &b)
    {
        return a + b;
    };
    std::cout << "==================================\n";
    // =======================================================

    auto func_print = []()
    {
        std::cout << "-------------------------\n";
    };
    func_print();

    typedef float (*FUNC_Fadd)(float, float);
    FUNC_Fadd     func_fadd = [](float x, float y)
    {
        return x + y;
    };
    std::cout << func_fadd(2.23, 1) << std::endl;

    return 0;
}

int ordinary_func_add(int x, int y)
{
    return x + y;
}
