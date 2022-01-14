/**
 * @file constant_nullptr.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++语言可用性增强之常量 nullptr, 主要替换 NULL
 * @version 0.1
 * @date 2022-01-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <type_traits>

void function_overload(char *);
void function_overload(int);

int main(int argc, char** argv)
{
    /* used decltype and std::is_same which are modern C++ syntax.
    decltype is used for type derivation, and std::is_same is used to compare the equality of the two types.
    
    NULL is different from 0 and nullptr.

    In a sense, traditional C++ treats NULL and 0 as the same thing, 
    depending on how the compiler defines NULL, and some compilers define NULL as ((void*)0) 
    Some will define it directly as 0.
    */
    if (std::is_same<decltype(NULL), decltype(0)>::value)
    {
        std::cout << "NULL == 0" << std::endl;
    }
    
    if (std::is_same<decltype(NULL), decltype((void*)0)>::value)
    {
        std::cout << "NULL == (void *)0" << std::endl;
    }
    
    if (std::is_same<decltype(NULL), std::nullptr_t>::value)
    {
        std::cout << "NULL == nullptr" << std::endl;
    }

    function_overload(0); /* will call function_overload(int) */
    // function_overload(NULL); /* does not compile because of overloading features in C++ to be confusing */
    function_overload(nullptr); /* will call function(char*) */
    
    return 0;
}


void function_overload(char *)
{
    std::cout << "function_overload(char *) is called\n";
}

void function_overload(int)
{
    std::cout << "function_overload(int) is called\n";
}