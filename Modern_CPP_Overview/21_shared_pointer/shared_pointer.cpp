/**
 * @file shared_pointer.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief smart pointer and memory management: std::shared_ptr
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <memory>

/* std::shared_ptr is a smart pointer that records how many shared_ptr points to an object, 
eliminating to call delete, which automatically deletes the object when the reference count becomes zero.

But not enough, because using std::shared_ptr still needs to be called with new, 
which makes the code a certain degree of asymmetry.

std::make_shared can be used to eliminate the explicit use of new, 
so std::make_shared will allocate the objects in the generated parameters. 
And return the std::shared_ptr pointer of this object type.

For example: */
void function_shared(std::shared_ptr<int> i)
{
    (*i)++;
}

int main(int argc, char **argv)
{
    // auto pointer = new int(10); /* un-recommend, no direct assignment */
    auto pointer = std::make_shared<int>(10); /* constructed a std::shared_ptr smart-pointer with std::make_shared */
    function_shared(pointer);
    std::cout << *pointer << std::endl; /* output: 11 */

    /* The shared_ptr will be destructed before leaving the scope. 

    std::shared_ptr can get the raw pointer through the get() method 
    and reduce the reference count by reset(). 
    And see the reference count of an object by use_count().
    */
    auto pointer2 = pointer;                                                     /* reference count +1 */
    auto pointer3 = pointer;                                                     /* reference count +1 */
    int *ptr = pointer.get();                                                    /* no increase of reference count */
    std::cout << *ptr << std::endl;
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   /* output: 3 */
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; /* output: 3 */
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; /* output: 3 */

    pointer2.reset();
    std::cout << "reset pointer2:" << std::endl;
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   /* output: 2 */
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; /* output: 0, pointer2 has reset */
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; /* output: 2 */

    pointer3.reset();
    std::cout << "reset pointer3:" << std::endl;
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   /* output: 1 */
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; /* output: 0 */
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; /* output: 0， pointer3 has reset */

    return 0;
}
