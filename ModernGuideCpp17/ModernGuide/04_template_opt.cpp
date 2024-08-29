/**
 * @file 04_template_opt.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/* 
 1. 模板的右尖括号
在泛型编程中,模板实例化有一个非常繁琐的地方,那就是连续的两个右尖括号（>>）
会被编译器解析成右移操作符,而不是模板参数表的结束.

 2. 默认模板参数
在C++98/03标准中，类模板可以有默认的模板参数;但是不支持函数的默认模板参数.
在C++11中添加了对函数模板默认参数的支持 .
 
*当所有模板参数都有默认参数时,函数模板的调用如同一个普通函数.
*但对于类模板而言,哪怕所有参数都有默认参数,在使用时也必须在模板名后跟随<>来实例化.
*另外：函数模板的默认模板参数在使用规则上和其他的默认参数也有一些不同,它没有必须写在参数表最后的限制,
*这样当默认模板参数和模板参数自动推导结合起来时, 书写就显得非常灵活了.
 
*/

#include <iostream>
#include <vector>

template<typename T>
class Base
{
public:
    void traversal(T &t)
    {
        auto it = t.begin();
        for (; it != t.end(); ++it)
        {
            std::cout << *it << " ";
        }
        std::cout << std::endl;
    }
};

// 默认模板参数
template<typename T = int, T t = 520>
class Test
{
public:
    void print()
    {
        std::cout << "current value: " << t << std::endl;
    }
};

// C++98/03不支持这种写法, C++11中支持这种写法
template<typename T = int>
void func(T t)
{
    std::cout << "current value: " << t << std::endl;
}

// 函数模板的默认模板参数在使用规则没有必须写在参数表最后的限制
template<typename R = int, typename N>
R func_default(N arg)
{
    return arg;
}

// 函数模板定义
template<typename T, typename U = char>
void func_infer(T arg1 = 100, U arg2 = 100)
{
    std::cout << "arg1: " << arg1 << ", arg2: " << arg2 << std::endl;
}

// ------------------------------------
int main(int argc, const char **argv)

{
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9};

    // 使用C++98/03标准来编译上边的这段代码
    // g++ .\04_template_opt.cpp -std=c++98
    /* 根据错误提示中描述模板的两个右尖括之间需要添加空格, 这样写起来就非常的麻烦,
    C++11改进了编译器的解析规则, 尽可能地将多个右尖括号（>）解析成模板参数结束符,
    方便我们编写模板相关的代码 */
    // Base<std::vector<int> > b;
    Base<std::vector<int>> b;
    b.traversal(v);

    // ==========================
    Test<> t;
    t.print();

    Test<int, 1024> t1;
    t1.print();

    func(100);

    // ?当默认模板参数和模板参数自动推导结合起来时
    // !函数返回值类型使用了默认的模板参数，函数的参数类型是自动推导出来的为int类型
    auto ret1 = func_default(520);
    std::cout << "\nreturn value-1: " << ret1 << std::endl;

    // !函数的返回值指定为double类型，函数参数是通过实参推导出来的，为double类型
    auto ret2 = func_default<double>(52.134);
    std::cout << "return value-2: " << ret2 << std::endl;

    // !函数的返回值指定为int类型，函数参数是通过实参推导出来的，为double类型
    auto ret3 = func_default<int>(52.134);
    std::cout << "return value-3: " << ret3 << std::endl;

    // !函数的参数为指定为int类型，函数返回值指定为char类型，不需要推导
    auto ret4 = func_default<char, int>(100);
    std::cout << "return value-4: " << ret4 << std::endl << std::endl;

    /* 当默认模板参数和模板参数自动推导同时使用时(优先级从高到低):
     *如果可以推导出参数类型则使用推导出的类型;
     *如果函数模板无法推导出参数类型, 那么编译器会使用默认模板参数;
     *如果无法推导出模板参数类型并且没有设置默认模板参数,编译器就会报错.
     !模板参数类型的自动推导是根据模板函数调用时指定的实参进行推断的,没有实参则无法推导.
     !模板参数类型的自动推导不会参考函数模板中指定的默认参数.
     */
    // 模板函数调用
    func_infer('a');
    func_infer(97, 'a');
    // func_infer(); //!编译报错

    return 0;
}
