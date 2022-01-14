/**
 * @file perfect_forwarding.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief rvalue reference; Perfect frowarding
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <utility>

/* the rvalue reference of a declaration is actually an lvalue.
   This creates problems for us to parameterize (pass)
*/
void reference(int &v)
{
    std::cout << " ---- lvalue reference ----" << std::endl;
}

void reference(int &&v)
{
    std::cout << " ---- rvalue reference ----" << std::endl;
}

template <typename T>
void pass(T &&v)
{
    std::cout << "         normal param passing: ";
    reference(v);
}

/* This is based on the reference collapsing rule: 
    In traditional C++, we are not able to continue to reference a reference type. 
    However, C++ has relaxed this practice with the advent of rvalue references,
    resulting in a reference collapse rule that allows us to reference references, both lvalue and rvalue.

    Perfect forwarding is based on the above rules. 
    The so-called perfect forwarding is to let us pass the parameters, 
    Keep the original parameter type (lvalue reference keeps lvalue reference, rvalue reference keeps rvalue reference). 
    To solve this problem, we should use std::forward to forward (pass) the parameters:

    std::forward is the same as std::move, and nothing is done. 
    std::move simply converts the lvalue to the rvalue. 
    std::forward is just a simple conversion of the parameters. 
    From the point of view of the phenomenon, std::forward<T>(v) is the same as static_cast<T&&>(v).
*/
template <typename T>
void pass_perfect(T &&v)
{
    std::cout << "         normal param passing: ";
    reference(v);

    std::cout << "      std::move param passing: ";
    reference(std::move(v));

    std::cout << "   std::forward param passing: ";
    reference(std::forward<T>(v));

    std::cout << "static_cat<T&&> param passing: ";
    reference(static_cast<T&&>(v));
}

int main(int argc, char **argv)
{
    std::cout << "rvalue pass: " << std::endl;
    pass(1);
    std::cout << "perfect forward: " << std::endl;
    pass_perfect(1);

    std::cout << "lvalue pass: " << std::endl;
    int lvalue = 1;
    pass(lvalue);
    std::cout << "perfect forward: " << std::endl;
    pass_perfect(lvalue);

    return 0;
}
