#include "12_10_4_stack.hpp"

Stack::Stack(int n)
{
    if (n < 0 || n > MAX)
        n = 10;

    ptr_items = new Item[n];

    size = n;
    top  = 0;
}

Stack::Stack(const Stack &st)
{
    size      = st.size;
    ptr_items = new Item[size];
    for (int i = 0; i < size; i++) ptr_items[i] = st.ptr_items[i];
    top = st.top;
}

Stack::~Stack()
{
    delete[] ptr_items;
}

bool Stack::is_empty() const
{
    return top == 0;
}

bool Stack::is_full() const
{
    return top == size;
}

bool Stack::push(const Item &item)
{
    if (top < size)
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

Stack &Stack::operator=(const Stack &st)
{
    if (this == &st)
        return *this;

    delete[] ptr_items;

    size      = st.size;
    top       = st.top;
    ptr_items = new Item[size];
    for (int i = 0; i < size; i++)
    {
        ptr_items[i] = st.ptr_items[i];
    }

    return *this;
}