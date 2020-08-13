#ifndef PEEKBACK_STACK_5_1_H
#define PEEKBACK_STACK_5_1_H

#include "Stack_5_1.hpp"

class Peekback_Stack : public Stack
{
public:
    Peekback_Stack(int capacity = 0) : _top(0)
    {
        if (capacity)
        {
            _stack.reserve(capacity);
        }
    }

    int size() const
    {
        return _stack.size();
    }

    bool empty() const
    {
        return ! _top;
    }

    bool full() const
    {
        return size() >= _stack.max_size();
    }

    int top() const
    {
        return _top;
    }

    void print(ostream &os = cout) const;

    bool pop(elemType &elem);
    bool push(const elemType &elem);
    bool peek(int, elemType &);

private:
    vector<elemType> _stack;
    int _top;
};

bool Peekback_Stack::pop(elemType &elem)
{
    if (empty())
    {
        return false;
    }
    elem = _stack[--_top];
    _stack.pop_back();

    return true;
}

bool Peekback_Stack::push(const elemType &elem)
{
    if (full())
    {
        return false;
    }
    _stack.push_back(elem);
    ++_top;

    return true;
}

void Peekback_Stack::print(ostream &os) const
{
    vector<elemType>::const_reverse_iterator rit = _stack.rbegin(), rend = _stack.rend();

    os << "\n\t";
    while (rit != rend)
    {
        os << *rit++ << "\n\t";
    }
    os << endl;
}

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

#endif  // PEEKBACK_STACK_5_1_H