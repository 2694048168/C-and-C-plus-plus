/**
 * @file auto_decltype.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 类型推断特性 auto and decltype keywords; tail type inference 尾部类型推断
 * @version 0.1
 * @date 2022-01-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <string>
#include <vector>

// since C++20, auto can even be used as function arguments
int add_auto(auto x, auto y)
{
    return x + y;
}

/* tail type inference 尾部类型推断 */
template <typename R, typename T, typename U>
R add_template(T x, U y)
{
    return x + y;
}

// want to like this, but not to be compiled.
/* 
template<typename T, typename U>
decltype(x+y) add_decltype(T x, U y)
{
    return x + y;
}
 */

template <typename T, typename U>
auto add_tail_type_inference(T x, U y) -> decltype(x + y)
{
    return x + y;
}

// Since C++14 directly derive the return value of a normal function.
template <typename T, typename U>
auto add_tail_type(T x, U y)
{
    return x + y;
}

// decltype(auto) since C++14
std::string lookup_1()
{
    std::string look_str_1 {"look up string(1)\n"};
    return look_str_1;
}

std::string look_str_2 {"look up string(2)\n"};
std::string& lookup_2()
{
    return look_str_2;
}

std::string look_up_a_string_1()
{
    return lookup_1();
}
std::string& look_up_a_string_2()
{
    return lookup_2();
}
/* With decltype(auto), we can let the compiler do this annoying parameter forwarding */
decltype(auto) look_up_string_1()
{
    return lookup_1();
}
decltype(auto) look_up_string_2()
{
    return lookup_2();
}


int main(int argc, char **argv)
{
    /* ---------------------------------------------------------- */
    // auto 主要用于复杂的类型推断(模板和迭代器等等)，体现代码的简洁性
    std::vector<int> vec{1, 2, 4, 5, 6, 9};
    for (auto iterator_vec = vec.begin(); iterator_vec != vec.end(); ++iterator_vec)
    {
        std::cout << *iterator_vec << ' ';
    }
    std::cout << std::endl;

    int x = 5;
    int y = 15;
    std::cout << add_auto(x, y) << std::endl;

    // Note: "auto" cannot be used to derive array types yet.
    auto arr = new auto(10);
    // auto auto_arr2[10] = {arr}; /* error: 'auto_arr2' declared as array of 'auto' */
    delete[] arr;

    /* ---------------------------------------------------------- */
    /* The decltype keyword is used to solve the defect that the auto keyword can only type the variable.
      decltype 关键字用于解决 auto 关键字只能推断变量的缺陷 
       Sometimes we may need to calculate the type of an expression
    */
    auto var_1 = 24;
    auto var_2 = 42;
    decltype(var_1 + var_2) z;
    z = var_1 + var_2;

    /* std::is_same<T, U> is used to determine whether the two types T and U are equal. */
    if (std::is_same<decltype(x), int>::value)
    {
        std::cout << "Type x == int" << std::endl;
    }

    if (!std::is_same<decltype(x), float>::value)
    {
        std::cout << "Type x != float" << std::endl;
    }

    if (std::is_same<decltype(x), decltype(z)>::value)
    {
        std::cout << "Type x == type z" << std::endl;
    }

    /* ---------------------------------------------------------- */
    /* tail type inference 尾部类型推断 */
    auto tail_type_inference = add_tail_type_inference<int, double>(1, 2.0);
    if (std::is_same<decltype(tail_type_inference), double>::value)
    {
        std::cout << "tail_type_inference is double.\n"
                  << tail_type_inference << std::endl;
    }

    auto tail_type = add_tail_type<double, int>(2.0, 40); /* C++14 */
    if (std::is_same<decltype(tail_type), double>::value)
    {
        std::cout << "tail_type is double.\n"
                  << tail_type << std::endl;
    }

    /* ---------------------------------------------------------- */
    /* decltype(auto) is a slightly more complicated use of C++14
        To understand it you need to know the concept of parameter forwarding (参数转发) in C++, 
        which will cover in detail in the Language Runtime Hardening
    */
   std::cout << "------decltype(auto)----------" << std::endl;
   std::cout << look_up_a_string_1 << std::endl;
   std::cout << look_up_a_string_2 << std::endl;
   std::cout << "----------------" << std::endl;
   std::cout << look_up_string_1 << std::endl;
   std::cout << look_up_string_2 << std::endl;

    return 0;
}
