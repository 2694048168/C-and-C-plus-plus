/**
 * @file noexcept_operator.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief noexcept and its operators
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

void may_throw()
{
    throw true;
}
auto non_block_throw = []
{
    may_throw();
};
void no_throw() noexcept
{
    return;
}

auto block_throw = []() noexcept
{
    no_throw();
};

int main(int argc, char const *argv[])
{
    std::cout << std::boolalpha
              << "may_throw() noexcept? " << noexcept(may_throw()) << std::endl
              << "no_throw() noexcept? " << noexcept(no_throw()) << std::endl
              << "lmay_throw() noexcept? " << noexcept(non_block_throw()) << std::endl
              << "lno_throw() noexcept? " << noexcept(block_throw()) << std::endl;

    try
    {
        may_throw();
    }
    catch (...)
    {
        std::cout << "exception captured from my_throw()" << std::endl;
    }

    try
    {
        non_block_throw();
    }
    catch (...)
    {
        std::cout << "exception captured from non_block_throw()" << std::endl;
    }

    try
    {
        block_throw();
    }
    catch (...)
    {
        std::cout << "exception captured from block_throw()" << std::endl;
    }
}
