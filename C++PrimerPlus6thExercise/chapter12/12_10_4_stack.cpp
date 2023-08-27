#include "12_10_4_stack.hpp"

Stack::Stack(int n) // create an empty stack
{
    top = 0;
}

Stack::~Stack() {}

bool Stack::is_empty() const
{
    return top == 0;
}

bool Stack::is_full() const
{
    return top == MAX;
}

bool Stack::push(const Item &item)
{
    if (top < MAX)
    {
        ptr_items[top++] = item;
        return true;
    }
    else
        return false;
}

bool Stack::pop(Item &item)
{
    if (top > 0)
    {
        item = ptr_items[--top];
        return true;
    }
    else
        return false;
}