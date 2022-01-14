/**
 * @file weak_pointer.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief smart pointer: std::weak_ptr
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <memory>

/* std::weak_ptr 
If you think about std::shared_ptr carefully, 
you will still find that there is still a problem that resources cannot be released. 
Look at the following example: */
class A;
class B;

class A
{
public:
    std::shared_ptr<B> pointer;
    ~A() { std::cout << "A was destroyed" << std::endl; }
};

class B
{
public:
    std::shared_ptr<A> pointer;
    ~B() { std::cout << "B was destroyed" << std::endl; }
};

int main(int argc, char **argv)
{
    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();
    a->pointer = b;
    b->pointer = a;
    /* The result is that A and B will not be destroyed. caused memory leak */

    /* The solution to this problem is to use the weak reference pointer std::weak_ptr, 
    which is a weak reference (compared to std::shared_ptr is a strong reference). 
    A weak reference does not cause an increase in the reference count. 
    When a weak reference is used, the final release process is shown in Figure 5.2:

    std::weak_ptr has no implemented * and -> operators, therefore it cannot operate on resources.
    std::weak_ptr allows us to check if a std::shared_ptr exists or not. 
    The expired() method of a std::weak_ptr returns false when the resource is not released; 
    Otherwise, it returns true. Furthermore, it can also be used for the purpose of obtaining std::shared_ptr, 
    which points to the original object. 
    The lock() method returns a std::shared_ptr to the original object 
    when the resource is not released,or nullptr otherwise. */

    return 0;
}
