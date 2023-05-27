/**
 * @file stack_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <stack>
#include <string>

/**
 * @brief Stacks are a type of container adaptors(LIFO) in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::stack<std::string> stack_1;
    stack_1.push("wei");
    stack_1.push("li");
    stack_1.push("jxufe");
    stack_1.push("software");

    std::cout << "the size of stack: " << stack_1.size() << std::endl;

    auto print = [&stack_1](const char* msg)
    {
        std::cout << msg;
        while (!stack_1.empty())
        {
            std::cout << stack_1.top() << " ";
            stack_1.pop();
        }
    };
    print("the elements of stack: ");

    std::cout << "\nthe size of stack: " << stack_1.size() << std::endl;

    return 0;
}