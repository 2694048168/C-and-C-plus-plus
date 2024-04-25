/**
 * @file 03_forwardLinkedLists.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief =====前向链表: 经典的基于指针的数据结构
 * 前向链表是由一系列元素组成的简单数据结构,
 * 每个元素都有一个指向下一个元素的指针, 链表中的最后一个元素持有 nullptr.
 * 在链表中插入元素是非常有效的, 而且元素在内存中可以是不连续的.
 * 
 */
struct Element
{
    Element *next{};

    void insert_after(Element *new_element)
    {
        new_element->next = next;
        next              = new_element;
    }

    char  prefix[2];
    short operating_number;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    Element trooper1, trooper2, trooper3;
    trooper1.prefix[0]        = 'T';
    trooper1.prefix[1]        = 'K';
    trooper1.operating_number = 421;
    trooper1.insert_after(&trooper2);
    trooper2.prefix[0]        = 'F';
    trooper2.prefix[1]        = 'N';
    trooper2.operating_number = 2187;
    trooper2.insert_after(&trooper3);
    trooper3.prefix[0]        = 'L';
    trooper3.prefix[1]        = 'S';
    trooper3.operating_number = 005;

    // 在每次迭代之前, 确保 cursor 指针不是 nullptr
    for (Element *cursor = &trooper1; cursor; cursor = cursor->next)
    {
        printf("storm-trooper %c%c-%d\n", cursor->prefix[0], cursor->prefix[1], cursor->operating_number);
    }

    /**
     * @brief ========== this指针
     * 请记住, 方法与类相关联, 类的实例是对象;
     * 当编写方法时, 有时需要访问当前对象, 也就是正在执行该方法的对象.
     * 在方法的定义中, 可以使用 this 指针访问当前对象,
     * 通常情况下,不需要使用 this, 因为在访问成员时 this 是隐式的.
     * 但有时, 可能需要消除歧义(这个方法参数的名称与成员变量重名时)
     * 
     */

    struct Element_
    {
        Element_ *next{};

        void insert_after(Element_ *new_element)
        {
            new_element->next = this->next;
            this->next        = new_element;
        }

        char  prefix[2];
        short operating_number;
    };

    struct ClockOfTheLongNow
    {
        bool set_year(int year)
        {
            if (year < 2019)
                return false;
            // 需要用 this 来消除成员和参数之间的歧义
            this->year = year;
            return true;
        }

    private:
        int year;
    };

    return 0;
}
