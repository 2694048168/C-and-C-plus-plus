#ifndef STACK_5_2_H
#define STACK_5_2_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

typedef string elemType;    

class Stack
{
public:
    Stack(int capacity = 0) : _top(0)
    {
        if (capacity)
        {
            _stack.reserve(capacity);
        }
    }

    virtual ~Stack(){};

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
    virtual bool peek(int, elemType &)
    {
        return false;
    }

protected:
    vector<elemType> _stack;
    int _top;
};

bool Stack::pop(elemType &elem)
{
    if (empty())
    {
        return false;
    }
    elem = _stack[--_top];
    _stack.pop_back();

    return true;
}

bool Stack::push(const elemType &elem)
{
    if (full())
    {
        return false;
    }
    _stack.push_back(elem);
    ++_top;

    return true;
}

void Stack::print(ostream &os) const
{
    vector<elemType>::const_reverse_iterator rit = _stack.rbegin(), rend = _stack.rend();

    os << "\n\t";
    while (rit != rend)
    {
        os << *rit++ << "\n\t";
    }
    os << endl;
}

ostream & operator<<(ostream &os, const Stack &rhs)
{
    rhs.print();
    return os;
}

#endif  // STACK_5_2_H