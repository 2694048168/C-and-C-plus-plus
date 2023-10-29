/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <functional>
#include <iostream>

/* Minimum function with any number of arguments

Write a function template that can take any number of arguments 
and returns the minimum value of them all, using operator < for comparison. 
Write a variant of this function template that can be parameterized 
with a binary comparison function to use instead of operator <.
------------------------------------------------------------------- */

/* Solution:
It is possible to write function templates that can take a variable number of arguments
using variadic function templates. 
For this, we need to implement compile-time recursion
(which is actually just calls through a set of overloaded functions). 
The following snippet shows how the requested function could be implemented:
---------------------------------------------------------------------------- */
template<typename T>
T minimum(const T a, const T b)
{
    return a < b ? a : b;
}

template<typename T, typename... Args>
T minimum(T a, Args... args)
{
    return minimum(a, minimum(args...));
}

template<typename T>
T maximum(const T a, const T b)
{
    return a > b ? a : b;
}

template<typename T, typename... Args>
T maximum(T a, Args... args)
{
    return maximum(a, maximum(args...));
}

/* In order to be able to use a user-provided binary comparison function, 
we need to write another function template. 
The comparison function must be the first argument 
because it cannot follow the function parameter pack. 
On the other hand, this cannot be an overload of the previous minimum function, 
but a function with a different name. 
The reason is that the compiler would not be able to
 differentiate between the template parameter lists
------------------------------------------------------- */
template<class Compare, typename T>
T minimum_compare(Compare functor, const T a, const T b)
{
    return functor(a, b) ? a : b;
}

template<class Compare, typename T, typename... Args>
T minimum_compare(Compare functor, T a, Args... args)
{
    return minimum_compare(functor, a, minimum_compare(functor, args...));
}

// -------------------------
int main(int argc, char **)
{
    auto min_value = minimum(5, 3, 1, 2);
    auto max_value = maximum(5, 3, 1, 2);
    std::cout << "[the minimum value]: " << min_value << '\n';
    std::cout << "[the maximum value]: " << max_value << '\n';

    // user-custom binary compare function
    auto min = minimum_compare(std::less<>(), 3, 2, 1, 0);
    std::cout << "[the minimum value]: " << min << '\n';

    auto max = minimum_compare(std::greater<>(), 3, 2, 1, 0);
    std::cout << "[the maximum value]: " << max << '\n';

    return 0;
}
