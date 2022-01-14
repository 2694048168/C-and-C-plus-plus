/**
 * @file extern_templates.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++ 特性之模板 templates
 *     C++ templates have always been a special art of the language, 
 * and templates can even be used independently as a new language. 
 * The philosophy of the template is to throw all the problems that 
 * can be processed at compile time into the compile time, 
 * and only deal with those core dynamic services at runtime, 
 * to greatly optimize the performance of the runtime. 
 * Therefore, templates are also regarded by many as one of the black magic of C++.
 * 
 * @version 0.1
 * @date 2022-01-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>

// step 1. extern templates
template class std::vector<bool>;          /* force instantiation */
extern template class std::vector<double>; /* should not instantiation in current file */

// step 2. the ">" std::vector<std::list<> >  or std::vector<std::list<>>;
// ">>" compiler always as a right shift operator
// 这个在 CUDA 高性能编程时常常遇到这种问题，分配多个计算核进行并行计算
std::vector<std::vector<int>> matrix;
template <bool T>
class MagicType
{
    bool magic = T;
};

// step 3. type alias templates
/* Templates are used to generate types.
 in traditional C++, "typedef" can define a new name for the type, 
 but there is no way to define a new name for the template.

 C++11 uses using to introduce the following form of writing, 
 and at the same time supports the same effect as the traditional typedef
*/
template <typename T, typename U>
class MagicType2
{
public:
    T dark;
    U magic;
};

// since C++ 11
typedef int (*process)(void *);
using NewProcess = int (*)(void *);
// type alias templates
template <typename T>
using TrueDarkMagic = MagicType2<std::vector<T>, std::string>;

// step 4. variadic templates
template <typename... Ts>
class Magic_Variadic;

template <typename Require, typename... Args>
class Magic_manually;

/* The variable length parameter template can also be directly adjusted to the template function. */
template <typename... Args>
void printf(const std::string &str, Args... args);

/** how to unpack the parameters, 
 * firstly, using "sizeof" to calculate the number of arguments.
 * secondly, the parameters are unpacked with classic processing methods,
 * method 1. Recursive template function;
 * method 2. Variable parameter template expansion in C++17; 
 * method 3. Initialize list expansion with the properties of Lambda expression in C++11;
 * 
 */
template <typename... Ts>
void magic_sizeof(Ts... args)
{
    std::cout << sizeof...(args) << std::endl;
}

// method 1. Recursive template function
template <typename T0>
void printf_recursive(T0 value)
{
    std::cout << value << std::endl;
}

template <typename T, typename... Ts>
void printf_recursive(T value, Ts... args)
{
    std::cout << value << std::endl;
    printf_recursive(args...);
}

// method 2. Variable parameter template expansion in C++17
template <typename T0, typename... T>
void printf_variable(T0 t0, T... t)
{
    std::cout << t0 << std::endl;
    if constexpr (sizeof...(t) > 0)
    {
        printf_variable(t...);
    }
}

// method 3. Initialize list expansion with the properties of Lambda expression in C++11
/* By initializing the list, (lambda expression, value)... will be expanded. 
    Due to the appearance of the comma expression, 
    the previous lambda expression is executed first, 
    and the output of the parameter is completed. 
    To avoid compiler warnings, we can explicitly convert std::initializer_list to void. 
*/
template<typename T, typename... Ts>
auto printf_init_lambda(T value, Ts... args)
{
    std::cout << value << std::endl;
    (void) std::initializer_list<T>{
        ([&args] {
            std::cout << args << std::endl;
        }(), value)...
    };
}

int main(int argc, char **argv)
{
    magic_sizeof();
    magic_sizeof(1);
    magic_sizeof(1, "weili");

    printf_recursive(1, 2, "weili", 1.1);

    return 0;
}
