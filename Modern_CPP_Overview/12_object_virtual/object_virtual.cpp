/**
 * @file object_virtual.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 显式的要求对虚函数进行重写: override and final keywords in C++11; Why???
 *      显式删除默认构造函数 Explicit delete default function; Why???
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

struct Base
{
    virtual void function_virtual(int);
};

struct SubClass : Base
{
    // override keyword explicitly overload, compiler check if the base function has such a virtual function
    virtual void function_virtual(int) override; /* legal */
    // virtual void function_virtual(float) override; /* illegal, no virtual function in super class */
};

/* final is to prevent the class from being continued to inherit 
and to terminate the virtual function to continue to be overloaded. */
class Base_final
{
public:
    virtual void function_final() final;
};

class SubClassFinal final : public Base_final
{
}; /* legal */

// class SubClassNoFinal : public SubClassFinal
// {
// }; /* illegal, SubClassFinal has final */

// class SubClass3 : public Base_final
// {
//     void function_final(); /* illegal, function_final has final */
// };

/* 
In traditional C++, if the programmer does not provide it, the compiler will default to generating
default constructors, copy constructs, assignment operators, and destructors for the object. 
Besides, C++ also defines operators such as new delete for all classes. 
This part of the function can be overridden when the programmer needs it.

C++11 provides a solution to the above requirements, 
allowing explicit declarations to take or reject functions that come with the compiler.  
*/
class Magic
{
public:
    Magic() = default;                        /* explicit let compiler use default constructor */
    Magic &operator=(const Magic &) = delete; /* explicit declare refuse constructor */
    Magic(int magic_number);
};

/* In traditional C++, enumerated types are not type-safe, and enumerated types are treated as integers, 
which allows two completely different enumerated types to be directly compared 
(although the compiler gives the check, but not all), 
Even the enumeration value names of different enum types in the same namespace cannot be the same, 
which is usually not what we want to see.
C++11 introduces an enumeration class and declares it using the syntax of enum class */
enum class new_enum : unsigned int
{
    value1,
    value2,
    value3 = 100,
    value4 = 100
};

// overload the "<<" operator to output for "new_enum"
template <typename T>
std::ostream &operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type &stream, const T &e)
{
    return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

int main(int argc, char **argv)
{
    if (new_enum::value3 == new_enum::value4) /* true */
    {
        std::cout << "new_enum::value3 == new_enum::value4" << std::endl;
    }

    // overload the "<<" operator to output for "new_enum"
    std::cout << new_enum::value3 << std::endl;

    return 0;
}
