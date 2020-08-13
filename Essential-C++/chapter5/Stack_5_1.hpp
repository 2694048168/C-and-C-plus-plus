#ifndef EXERCISE_5_1_H
#define EXERCISE_5_1_H

#include <string>
#include <iostream>
#include <vector>

using namespace std;

typedef string elemType;

class Stack
{
public:
    virtual ~Stack() {};
    virtual bool pop(elemType &) = 0;
    virtual bool push(const elemType &) = 0;
    virtual bool peek(int index, elemType &) = 0;

    virtual int top() const = 0;
    virtual int size() const = 0;

    virtual bool empty() const = 0;
    virtual bool full() const = 0;
    virtual void print(ostream & = cout) const = 0;
};

ostream & operator<<(ostream &os, const Stack &rhs)
{
    rhs.print();
    return os;
}

#endif  // EXERCISE_5_1_H