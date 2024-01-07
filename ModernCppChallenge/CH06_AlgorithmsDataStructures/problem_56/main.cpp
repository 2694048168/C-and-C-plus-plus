/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Select algorithm
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>


/**
 * @brief Select algorithm
 * 
 * Write a function that, given a range of values and a projection function, 
 * transforms each value into a new one and returns a new range with the selected values. 
 * For instance, if you have a type book that has an id, title, and author, 
 * and have a range of such book values, 
 * it should be possible for the function to select only the title of the books.
 * 
 * The select() function that you have to implement takes an std::vector<T> 
 * as an input argument and a function of type F and returns a std::vector<R> as the result,
 * where R is the result of applying F to T. We could use std::result_of() to 
 * deduce the return type of an invoke expression at compile time. 
 * Internally, the select() function should use std::transform() 
 * to iterate over the elements of the input vector, apply function f 
 * to each element, and insert the result in an output vector.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename T, typename A, typename F,
         typename R = typename std::decay<typename std::result_of<
             typename std::decay<F>::type &(typename std::vector<T, A>::const_reference)>::type>::type>
std::vector<R> select(const std::vector<T, A> &c, F &&f)
{
    std::vector<R> v;
    std::transform(std::cbegin(c), std::cend(c), std::back_inserter(v), std::forward<F>(f));

    return v;
}

struct book
{
    int         id;
    std::string title;
    std::string author;
};

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<book> books{
        {101,        "The C++ Programming Language", "Bjarne Stroustrup"},
        {203,                "Effective Modern C++",      "Scott Meyers"},
        {404, "The Modern C++ Programming Cookbook",    "Marius Bancila"}
    };

    auto titles = select(books, [](const book &b) { return b.title; });

    for (const auto &title : titles)
    {
        std::cout << title << std::endl;
    }

    return 0;
}
