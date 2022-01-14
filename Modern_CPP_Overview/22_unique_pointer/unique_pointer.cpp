/**
 * @file unique_pointer.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief smart pointer: std::unique_ptr
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <memory>

/* std::unique_ptr is an exclusive smart pointer 
that prohibits other smart pointers from sharing the same object, 
thus keeping the code safe.

Since it is monopolized, in other words, it cannot be copied. 
However, we can use std::move to transfer it to other unique_ptr, 
for example:
*/
struct Function
{
    Function() { std::cout << "Function::Function" << std::endl; }
    ~Function() { std::cout << "Function::~Function" << std::endl; }
    void function() { std::cout << "Function::function" << std::endl; }
};

void func(const Function &)
{
    std::cout << "func(const Function&)" << std::endl;
}

int main(int argc, char **argv)
{
    std::unique_ptr<int> pointer = std::make_unique<int>(10); /* std::make_unique was introduced in C++14 */
    // std::unique_ptr<int> pointer2 = pointer;                  /* illegal */

    std::unique_ptr<Function> ptr1(std::make_unique<Function>());
    if (ptr1) /* ptr1 is not empty, and then prints */
    {
        ptr1->function();
    }

    {
        std::unique_ptr<Function> ptr2(std::move(ptr1));
        func(*ptr2); /* ptr2 is not empty, prints */
        if (ptr2)
        {
            ptr2->function();
        }

        if (ptr1) /* ptr1 is empty, no prints */
        {
            ptr1->function();
        }

        ptr1 = std::move(ptr2);
        if (ptr2) /* ptr2 is empty, no prints */
        {
            ptr2->function();
        }
        std::cout << "ptr2 was destroyed here in this scope." << std::endl; 
    }

    /* ptr1 is not empty,  prints */
    if (ptr1)
    {
        ptr1->function();
    }
    
    // Function instance will be destroyed when leaving the scope
    return 0;
}
