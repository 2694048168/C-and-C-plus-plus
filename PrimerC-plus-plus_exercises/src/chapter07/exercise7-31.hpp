/* exercise 7-31
** 练习7.31: 定义一对类X和Y，其中X 包含一个指向 Y 的指针，而Y包含一个类型为 X 的对象。
**
*/

#ifndef EXERCISE7_31_H
#define EXERCISE7_31_H

class Y;

class X 
{
    Y* y = nullptr;
};

class Y 
{
    X x;
};


#endif // EXERCISE7_31_H
