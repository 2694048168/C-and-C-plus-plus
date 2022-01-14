/**
 * @file magic_template.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++ templates is a black magic features. Fold expression and Non-type template parameter deduction
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

// C++17, this feature of the variable length parameter is further brought to the expression
// type template parameters.
template <typename... T>
auto sum_arbitrary(T... t)
{
    return (t + ...);
}

// Non-type template parameter deduction
template <typename T, int BufSize>
class buffer_t
{
public:
    T &alloc();
    void free(T &item);

private:
    T data[BufSize];
};

// C++17 compiler assist in the completion of specific types of derivation with auto
template <auto value>
void function_assist()
{
    std::cout << value << std::endl;
    return;
}

int main(int argc, char **argv)
{
    std::cout << sum_arbitrary(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) << std::endl;

    // non-type template parameter
    buffer_t<int, 100> buf; /* 100 as template parameter */

    // C++17
    function_assist<10>(); /* value as int */

    return 0;
}
