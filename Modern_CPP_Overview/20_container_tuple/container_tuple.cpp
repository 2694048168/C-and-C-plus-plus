/**
 * @file container_tuple.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief std::pair and std::tuple; Runtime Indexing; Merge and Iteration
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <tuple>
#include <variant>

/* std::pair
Programmers who have known Python should be aware of the concept of tuples. 
Looking at the containers in traditional C++, 
except for std::pair there seems to be no ready-made structure to store different types of data 
(usually we will define the structure ourselves). 
But the flaw of std::pair is obvious, only two elements can be saved. */

/* step 1. Basic Operations
There are three core functions for the use of tuples:
1. std::make_tuple: construct tuple
2. std::get: Get the value of a position in the tuple
3. std::tie: tuple unpacking */
auto get_student(int id)
{
    if (id == 0)
    {
        return std::make_tuple(3.8, 'A', "John");
    }
    if (id == 1)
    {
        return std::make_tuple(2.9, 'C', "Jack");
    }
    if (id == 2)
    {
        return std::make_tuple(4.9, 'A', "Wei Li");
    }

    /* it is not allowed to return 0 directly, "return 0;"
    return type is std::tuple<double, char, std::string> */
    return std::make_tuple(0.0, 'D', "NULL");
}

// ---- step 2. Runtime Indexing ----
template <size_t n, typename... T>
constexpr std::variant<T...> _tuple_index(const std::tuple<T...> &tpl, size_t i)
{
    if constexpr (n >= sizeof...(T))
        throw std::out_of_range("out of range.");
    if (i == n)
        return std::variant<T...>{std::in_place_index<n>, std::get<n>(tpl)};
    return _tuple_index<(n < sizeof...(T) - 1 ? n + 1 : 0)>(tpl, i);
}

template <typename... T>
constexpr std::variant<T...> tuple_index(const std::tuple<T...> &tpl, size_t i)
{
    return _tuple_index<0>(tpl, i);
}

template <typename T0, typename... Ts>
std::ostream &operator<<(std::ostream &s, std::variant<T0, Ts...> const &v)
{
    std::visit([&](auto &&x)
               { s << x; },
               v);
    return s;
}

// ---- step 3. Merge and Iteration ----
template <typename T>
auto tuple_len(T &tpl)
{
    return std::tuple_size<T>::value;
}

int main(int argc, char **argv)
{
    // ---- step 1. Basic Operations ----
    int student_id = 2;
    auto student = get_student(student_id);
    std::cout << "ID: " << student_id << ", "
              << "GPA: " << std::get<0>(student) << ", "
              << "Grade: " << std::get<1>(student) << ", "
              << "Name: " << std::get<2>(student) << '\n';

    double gpa;
    char grade;
    std::string name;
    // unpack tuples
    std::tie(gpa, grade, name) = get_student(1);
    std::cout << "ID: 1, "
              << "GPA: " << gpa << ", "
              << "Grade: " << grade << ", "
              << "Name: " << name << '\n';

    /* std::get In addition to using constants to get tuple objects, 
    C++14 adds usage types to get objects in tuples: */
    std::tuple<std::string, double, double, int> tuple_new("WeiLi", 4.9, 9.9, 42);
    std::cout << std::get<std::string>(tuple_new) << std::endl;
    // std::cout << std::get<double>(tuple_new) << std::endl; /* illegal, runtime error */
    std::cout << std::get<3>(tuple_new) << std::endl;

    // ---- step 2. Runtime Indexing ----
    /* If you think about it, you might find the problem with the above code. 
    std::get<> depends on a compile-time constant, so the following is not legal: */
    // int index_tuple = 1;
    // std::get<index_tuple>(tuple_new);

    /* So what do you do? 
    The answer is to use std::variant<> (introduced by C++ 17) to provide type template parameters for variant<> 
    You can have a variant<> to accommodate several types of variables provided 
    (in other languages, such as Python/JavaScript, etc., as dynamic types) */
    int i = 0;
    std::cout << tuple_index(tuple_new, i) << std::endl;

    // ---- step 3. Merge and Iteration ----
    // merge two tuples with std::tuple_cat

    auto new_tuple = std::tuple_cat(get_student(2), std::move(tuple_new));

    /* You can immediately see how quickly you can traverse a tuple? 
    But we just introduced how to index a tuple by a very number at runtime, 
    then the traversal becomes simpler. First, we need to know the length of a tuple, 
    which can: */
    for (size_t i = 0; i != tuple_len(new_tuple); ++i)
    {
        /* runtime indexing */
        std::cout << tuple_index(new_tuple, i) << " ";
    }
    std::cout << "\n";

    return 0;
}
