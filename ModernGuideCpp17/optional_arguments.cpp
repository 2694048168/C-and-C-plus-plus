/**
 * @file optional_arguments.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-01
 * 
 * @copyright Copyright (c) 2023
 * 
 * the std::optional feature since C++17.
 * 
 */

#include <functional>
#include <iostream>
#include <optional>
#include <string>


// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b)
{
    if (b)
        return "Godzilla";
    return {};
}

// std::nullopt can be used to create any (empty) std::optional
auto create_better(bool b)
{
    return b ? std::optional<std::string>{"Godzilla"} : std::nullopt;
}

// std::reference_wrapper may be used to return a reference
auto create_ref(bool b)
{
    static std::string value = "Godzilla";
    return b ? std::optional<std::reference_wrapper<std::string>>{value} : std::nullopt;
}

void foo(const int i, std::optional<double> f, std::optional<bool> b) {}

/**
 * @brief Allow argument values to be omitted when calling a function.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* The type std::optional (from the Library Fundamentals TS). 
     This allows the value of those arguments to be omitted, 
     where std::nullopt represents no value.

     This approach is more expressive than using pointers and nullptr. 
     A related technique is the use of default arguments, 
     which allow arguments to be omitted entirely, 
     but only from the end of the argument list.
    -----------------------------------------------  */
    foo(42, 1.0, true);

    foo(42, std::nullopt, true);

    foo(42, 1.00, std::nullopt);

    foo(42, std::nullopt, std::nullopt);

    std::cout << "-----------------------------" << std::endl;
    std::cout << "create(false) returned " << create(false).value_or("empty") << '\n';

    // optional-returning factory functions are usable as conditions of while and if
    if (auto str = create_better(true))
        std::cout << "create_better(true) returned " << *str << '\n';

    if (auto str = create_ref(true))
    {
        // using get() to access the reference_wrapper's value
        std::cout << "create_ref(true) returned " << str->get() << '\n';
        str->get() = "Mothra";
        std::cout << "modifying it changed it to " << str->get() << '\n';
    }

    return 0;
}