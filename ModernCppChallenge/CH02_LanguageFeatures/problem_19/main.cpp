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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <vector>

/* Adding a range of values to a container
Write a general-purpose function that can add any number of elements 
to the end of a container that has a method push_back(T&& value).
------------------------------------------------------------------- */

/* Solution:
Writing functions with any number of arguments is possible
 using variadic function templates. 
The function should have the container as the first parameter, 
followed by a variable number of arguments representing the values to be
 added at the back of the container. 
However, writing such a function template can be significantly simplified 
using fold expressions. Such an implementation is shown here:
--------------------------------------------------------------- */
template<typename Container, typename... Args>
void push_back(Container &container, Args &&...args)
{
    (container.push_back(args), ...);
}

// -------------------------
int main(int argc, char **)
{
    std::vector<int> vec;
    push_back(vec, 1, 2, 3, 4);
    std::copy(std::begin(vec), std::end(vec), std::ostream_iterator<int>(std::cout, " "));

    std::cout << "\n-----------------------\n";

    std::list<int> l;
    push_back(l, 1, 2, 3, 4);
    std::copy(std::begin(l), std::end(l), std::ostream_iterator<int>(std::cout, " "));

    return 0;
}
