/**
 * @file 17_using_alias.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++中using用于声明命名空间,使用命名空间也可以防止命名冲突.
 * 在程序中声明了命名空间之后,就可以直接使用命名空间中的定义的类了.
 * *在C++11中赋予了using新的功能,让C++变得更年轻,更灵活.
 * 
 * 1. 定义别名
 * C++11中规定了一种新的方法,使用别名声明(alias declaration)来定义类型的别名,即使用using.
 * !通过 typedef 重定义一个类型, 被重定义的类型并不是一个新的类型,仅仅只是原有的类型取了一个新的名字.
 * 
 * 2. 模板的别名
 * 
 * 
 * 
 * 
 * 
 * 
 */

#include <iostream>
#include <map>

/* 如果不是特别熟悉函数指针与typedef,第一眼很难看出func_ptr其实是一个别名,
其本质是一个函数指针,指向的函数返回类型是int,函数参数有两个分别是int，double类型.
*使用using定义函数指针别名的写法看起来就非常直观了,
把别名的名字强制分离到了左边,而把别名对应的实际类型放在了右边,比较清晰,可读性比较好. */
// 使用typedef定义函数指针
typedef int (*func_ptr)(int, double);

// 使用using定义函数指针
using func_ptr1 = int (*)(int, double);

int addNum(int x, double y)
{
    return x + y;
}

int subNum(int x, double y)
{
    return x - y;
}

// =============================
template<typename T>
using my_map = std::map<int, T>;

template<typename T>
void traversal(my_map<T> a)
{
    for (auto it = a.begin(); it != a.end(); ++it)
    {
        std::cout << it->first << " ---> " << it->second << "\n";
    }
    std::cout << std::endl;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    func_ptr  add_func1 = addNum;
    func_ptr1 add_func2 = addNum;
    std::cout << "func_ptr = " << add_func1(42, 12.23) << std::endl;
    std::cout << "func_ptr1 = " << add_func2(42, 12.23) << std::endl;

    func_ptr  sub_func1 = subNum;
    func_ptr1 sub_func2 = subNum;
    std::cout << "func_ptr = " << sub_func1(42, 12.23) << std::endl;
    std::cout << "func_ptr1 = " << sub_func2(42, 12.23) << std::endl;

    // map的value指定为string类型
    my_map<std::string> m_str;
    m_str.insert(std::make_pair(1, "Ithaca"));
    m_str.insert(std::make_pair(2, "ace"));
    traversal(m_str);

    // map的value指定为int类型
    my_map<int> m_integer;
    m_integer.insert(std::make_pair(1, 100));
    m_integer.insert(std::make_pair(2, 200));
    traversal(m_integer);

    /* using语法和typedef一样,并不会创建出新的类型,
    它们只是给某些类型定义了新的别名.
    using相较于typedef的优势在于定义函数指针别名时看起来更加直观,并且可以给模板定义别名. */

    return 0;
}
