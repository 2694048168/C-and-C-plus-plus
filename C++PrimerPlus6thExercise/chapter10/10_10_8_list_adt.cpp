#include "10_10_8_list_adt.hpp"

#include <iostream>

ListADT::ListADT(const Item arr[], const unsigned size)
{
    for (size_t i = 0; i < size; ++i)
    {
        this->items[i] = arr[i];
    }
}

void ListADT::list_modify(Item val, unsigned idx)
{
    this->items[idx] = val;
}

bool ListADT::is_empty()
{
    // TODO 这里判断的标准需要进一步修正
    if (this->items[0])
    {
        return false;
    }
    return true;
}

bool ListADT::is_full()
{
    // TODO 这里判断的标准需要进一步修正
    if (!this->items[MAX - 1])
    {
        return false;
    }
    return true;
}

void ListADT::show_item() const
{
    std::cout << "The value of Item: [ ";
    for (size_t i = 0; i < MAX; ++i)
    {
        std::cout << items[i] << " ";
    }
    std::cout << "]\n--------------------------\n";
}

void ListADT::item_op(unsigned idx, void (*op_func)(Item &))
{
    op_func(this->items[idx]);
}

void ListADT::item_op(void (*op_func)(Item &))
{
    for (size_t idx = 0; idx < MAX; ++idx)
    {
        op_func(this->items[idx]);
    }
}