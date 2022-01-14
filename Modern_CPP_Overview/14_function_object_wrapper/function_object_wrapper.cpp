/**
 * @file function_object_wrapper.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Function Object Wrapper; std::function; std::bind and std::placeholder
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <functional> /* generic and polymorphic function wrapper */

/* The essence of a Lambda expression is an object of a class type (called a closure type) that is similar
    to a function object type (called a closure object). When the capture list of a Lambda expression is
    empty, the closure object can also be converted to a function pointer value for delivery, for example: 
*/
using function_ptr = void(int); /* function pointer */
void functional(function_ptr func_ptr)
{
    std::cout << "This is Function pointer: ";
    func_ptr(1);
}

/* std::function
    C++11 std::function is a generic, polymorphic function wrapper whose instances can store, 
    copy, and call any target entity that can be called. 
    When we have a container for functions, 
    we can more easily handle functions and function pointers as objects. 
*/
int function_wrapper(int para)
{
    return para;
}

/* std::bind and std::placeholder
    std::bind is used to bind the parameters of the function call. 
    It solves the requirement that we may not always be able to get all the parameters of a function at one time. 
    Through this function, we can Part of the call parameters are bound to the function in advance to become a new object, and then complete the call after the parameters are complete. e.g: 
*/
int function_bind(int a, int b, int c)
{
    return a + b + c;
}

int main(int argc, char **argv)
{
    auto func_lambda = [](int value)
    {
        std::cout << "This is Lambda Expression: ";
        std::cout << value << std::endl;
    };

    functional(func_lambda); /* call by function pointer */
    func_lambda(1);          /* call by lambda expression */

    // std::function wraps a function that take int parameter and returns int value.
    std::function<int(int)> func_wrap = function_wrapper;

    int important = 10;
    std::function<int(int)> func_lambda_wrap = [&](int value) -> int
    {
        return 1 + value + important;
    };

    std::cout << "func_wrap: " <<  func_wrap(10) << std::endl;
    std::cout << "func_lambda_wrap: " << func_lambda_wrap(10) << std::endl;

    // std::bind and std::placeholder
    /* bind parameter 1, 2 on function function_bind, 
    and use std::placeholder::_1 as placeholder for the first parameter. 
    
    Tip: Note the magic of the auto keyword. Sometimes we may not be familiar with the return type of a function, 
    but we can circumvent this problem by using auto.
    */
    auto bind_function = std::bind(function_bind, std::placeholders::_1, 1, 2);
    /* when call "bind_function", we only need one param left. */
    std::cout << "bind_function(1): " << bind_function(1) << std::endl;
    std::cout << "bind_function(39): " << bind_function(39) << std::endl;

    return 0;
}
