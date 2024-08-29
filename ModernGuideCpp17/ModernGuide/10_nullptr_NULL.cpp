/**
 * @file 10_nullptr_NULL.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++程序开发中,为了提高程序的健壮性,一般会在定义指针的同时完成初始化操作,
 *  或者在指针的指向尚未明确的情况下,都会给指针初始化为NULL, 
 * 避免产生野指针（没有明确指向的指针，操作也这种指针极可能导致程序发生异常）.
 * C++98/03 标准中，将一个指针初始化为空指针的方式有 2 种：
 *  --- char *ptr = 0;
 *  --- char *ptr = NULL;
 * 也就是说如果源码是C++程序NULL就是0，如果是C程序NULL表示(void*)0。那么为什么要这样做呢？ 
 * 是由于 C++ 中，void * 类型无法隐式转换为其他类型的指针，此时使用 0 代替 ((void *)0)，
 * 用于解决空指针的问题。这个0（0x0000 0000）表示的就是虚拟地址空间中的0地址，这块地址是只读的.
 * 
 * !C++ 中将 NULL 定义为字面常量 0,并不能保证在所有场景下都能很好的工作,比如函数重载时,NULL 和 0 无法区分.
 * 出于兼容性的考虑,C++11 标准并没有对 NULL 的宏定义做任何修改,
 * 而是另其炉灶引入了一个新的关键字nullptr,
 * nullptr 专用于初始化空类型指针，不同类型的指针变量都可以使用 nullptr 来初始化.
 * *编译器会分别将 nullptr 隐式转换成 int*、char* 以及 double* 指针类型.
 * ?在 C++11 标准下相比 NULL 和 0,使用 nullptr 初始化空指针可以令编写的程序更加健壮.
 *
 */

#include <iostream>

void func(char *p)
{
    std::cout << "void func(char *p)" << std::endl;
}

void func(int p)
{
    std::cout << "void func(int p)" << std::endl;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    // func(NULL); // 想要调用重载函数 void func(char *p)
    func(nullptr); // 想要调用重载函数 void func(char *p)
    func(250);     // 想要调用重载函数 void func(int p)

    return 0;
}
