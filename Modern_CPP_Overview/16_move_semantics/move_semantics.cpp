/**
 * @file move_semantics.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief rvalue std::move; Move semantics
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <utility> /* std::move */
#include <vector>
#include <string>

/* Traditional C++ has designed the concept of copy/copy for class objects 
    through copy constructors and assignment operators, but to implement the movement of resources, 
    The caller must use the method of copying and then destructing first, 
    otherwise, you need to implement the interface of the mobile object yourself. 
    
    Imagine moving your home directly to your new home instead of copying everything (rebuy) to your new home. 
    Throwing away (destroying) all the original things is a very anti-human thing.

    Traditional C++ does not distinguish between the concepts of “mobile” and “copy”, 
    resulting in a large amount of data copying, wasting time and space. 
    The appearance of rvalue references solves the confusion of these two concepts, 
for example: 
*/
class Basic_Class
{
public:
    int *pointer;
    Basic_Class() : pointer(new int(1))
    {
        std::cout << "construct " << pointer << std::endl;
    }

    Basic_Class(Basic_Class &&basic_class) : pointer(basic_class.pointer)
    {
        basic_class.pointer = nullptr;
        std::cout << "move " << pointer << std::endl;
    }

    ~Basic_Class()
    {
        std::cout << "destruct " << pointer << std::endl;
        delete pointer;
    }
};

// avoid compiler optimization
Basic_Class return_rvalue(bool test)
{
    Basic_Class a, b;
    if (test)
    {
        return a; /* equal to static_cast<Basic_Class&&>(a) */
    }
    else
    {
        return b;
    }
}

int main(int argc, char **argv)
{
    // 仔细查看资源所在的地址，copy 而不是 move 导致资源的浪费和时间的消耗
    Basic_Class obj = return_rvalue(false);
    std::cout << "obj: " << obj.pointer << std::endl;
    std::cout << *obj.pointer << std::endl;

    /* This avoids meaningless copy constructs and enhances performance. 
    Let’s take a look at an example involving a standard library: */
    std::cout << "-----------------" << std::endl;
    std::string str = "hello world.";
    std::vector<std::string> vec;

    // use push_back(const T&), copy
    vec.push_back(str);
    std::cout << "str: " << str << std::endl;

    // use push_back(const T&&), no copy
    /* the string will be moved to vector, and therefore std::move can reduce copy cost */
    vec.push_back(std::move(str)); /* str is empty now */
    std::cout << "str: " << str << std::endl;

    return 0;
}
