/**
 * @file 10_10_8_list_adt.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __LIST_ADT_HPP__
#define __LIST_ADT_HPP__

// ListADT 可以存储的数据类型, 抽象为 Item
typedef int Item;

// typedef float Item;

class ListADT
{
private:
    enum
    {
        MAX = 10
    };

    Item items[MAX] = {0};

public:
    ListADT() = default;

    ListADT(const Item arr[], const unsigned size);
    void list_modify(Item val, unsigned idx = 0);
    bool is_empty();
    bool is_full();
    void show_item() const;

    /* 函数指针说的就是一个指针，但这个指针指向的函数，不是普通的基本数据类型或者类对象。
      指向函数的指针包含了函数的地址，可以通过它来调用函数。
      声明格式：类型说明符 (*函数名)(参数) ----> int (*func)(int a, int b);
    ----------------------------------------------------------------- */
    void item_op(unsigned idx, void (*op_func)(Item &));
    void item_op(void (*op_func)(Item &));
};

#endif // !__LIST_ADT_HPP__