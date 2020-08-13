#ifndef PEEKBACK_STACK_5_2_H
#define PEEKBACK_STACK_5_2_H

#include "Stack_5_2.hpp"

class Peekback_Stack : public Stack
{
public:
    Peekback_Stack(int capacity = 0) : Stack(capacity) {}

    virtual bool peek(int index, elemType &elem);
};


bool Peekback_Stack::peek(int index, elemType &elem)
{
    if (empty())
    {
        return false;
    }

    if (index < 0 || index >= size())
    {
        return false;
    }

    elem = _stack[index];

    return true;
}

#endif  // PEEKBACK_STACK_5_2_H