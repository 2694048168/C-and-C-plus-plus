/**
 * @file lambda_expression.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++11 新特性 lambda expression; basics and generic lambda expression;
 * Lambda expressions are one of the most important features in modern C++, 
 * and Lambda expressions provide a feature like anonymous functions. 
 * Anonymous functions are used when a function is needed, but you don’t want to use a name to call a function. 
 * There are many, many scenes like this. So anonymous functions are almost standard in modern programming languages.
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <memory>  /* std::make_unique */
#include <utility> /* std::move */

/* step 1. Basics
    The basic syntax of a Lambda expression in as follows:
    [capture list] (parameter list) mutable(optional) exception attribute -> return type
    {
        // function body;
    }

    the capture list is also divided into the following types:
    1. Value capture
    2. Reference capture
    3. Implicit capture
    4. Expression capture
 */
void lambda_value_capture()
{
    int value = 1;
    auto copy_value = [value]
    {
        return value;
    };

    value = 100;
    auto stored_value = copy_value();
    std::cout << "stored_value = " << stored_value << std::endl;
    std::cout << "value = " << value << std::endl;
    /* At this moment, stored_value == 1, and value == 100.
    Because copy_value (lambda expression) has copied when its was created. */
}

void lambda_reference_capture()
{
    int value = 2;
    auto copy_value = [&value]
    {
        return value;
    };

    value = 200;
    auto stored_value = copy_value();
    std::cout << "stored_value = " << stored_value << std::endl;
    std::cout << "value = " << value << std::endl;
    /* At this moment, stored_value == 200, and value == 200.
    Because copy_value (lambda expression) store reference. */
}

/* Implicit capture
capture provides the ability for lambda expression to use external values.
The four most common forms of capture lists can be:
    1. [] empty capture list;
    2. [name1, name2, ...] captures a series of variables;
    3. [&] reference capture, let the compiler deduce the reference list by itself;
    4. [=] value capture, let the compiler deduce the value list by itself.
 */

/* Expression capture
capture methods capture the lvalue and not capture the rvalue.
C++14 allows the capture of rvalue with arbitrary expression. 
 */
void lambda_expression_capture()
{
    auto important = std::make_unique<int>(1);
    /* important is an exclusive pointer that cannot be caught by value capture using =. 
    At this time we need to transfer it to the rvalue and initialize it in the expression. */
    auto add = [v1 = 1, v2 = std::move(important)](int x, int y) -> int
    {
        return x + y + v1 + (*v2);
    };

    std::cout << add(3, 4) << std::endl;
}

/* step 2. Generic Lambda
    In C++11, auto keyword cannot be used in the parameter list 
    because it would conflict with the functionality of the template. 
    Starting with C++14. The formal parameters of the Lambda function can use the auto keyword to generate generic meanings
 */
void lambda_generic()
{
    auto generic = [](auto x, auto y)
    {
        return x + y;
    };

    std::cout << generic(12, 30) << std::endl;
    std::cout << generic(11.5, 30.0) << std::endl;
}

int main(int argc, char **argv)
{
    lambda_value_capture();
    lambda_reference_capture();
    lambda_expression_capture();
    lambda_generic();

    return 0;
}
