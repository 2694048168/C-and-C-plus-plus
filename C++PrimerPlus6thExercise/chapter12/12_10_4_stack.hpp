/**
 * @file 12_10_4_stack.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __STACK_HPP__
#define __STACK_HPP__

// class declaration for the stack ADT
typedef unsigned long Item;

class Stack
{
private:
    enum
    {
        MAX = 10
    }; // constant specific to class

    Item *ptr_items; // holds stack items
    int   size;      // number of elements in stack
    int   top;       // index for top stack item

public:
    // Stack();
    Stack(int n = MAX); // creates stack with n elements
    Stack(const Stack &st);
    ~Stack();

    bool is_empty() const;
    bool is_full() const;

    // push() returns false if stack already is full, true otherwise
    bool push(const Item &item); // add item to stack

    // pop() returns false if stack already is empty, true otherwise
    bool pop(Item &item); // pop top into item

    Stack &operator=(const Stack &st);
};

#endif // !__STACK_HPP__