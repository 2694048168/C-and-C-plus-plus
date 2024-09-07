/**
 * @file 25_unique_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 初始化
 * std::unique_ptr是一个独占型的智能指针,它不允许其他的智能指针共享其内部的指针,
 * 可以通过它的构造函数初始化一个独占智能指针对象,但是不允许通过赋值将一个unique_ptr赋值给另一个unique_ptr.
 * ?std::unique_ptr不允许复制,但是可以通过函数返回给其他的std::unique_ptr,
 * ?还可以通过std::move来转译给其他的std::unique_ptr,这样原始指针的所有权就被转移了,这个原始指针还是被独占的.
 * 
 * *使用reset方法可以让unique_ptr解除对原始内存的管理，也可以用来初始化一个独占的智能指针
 * 
 * 2. 删除器
 * std::unique_ptr指定删除器和std::shared_ptr指定删除器是有区别的,
 * std::unique_ptr指定删除器的时候需要确定删除器的类型,所以不能像std::shared_ptr那样直接指定删除器
 * 
 */

#include <functional>
#include <iostream>
#include <memory>

std::unique_ptr<int> func()
{
    return std::unique_ptr<int>(new int(520));
}

// -------------------------------------
int main(int argc, const char **argv)
{
    // 通过构造函数初始化对象
    std::unique_ptr<int> ptr1(new int(42));
    // !error, 不允许将一个unique_ptr赋值给另一个unique_ptr
    // std::unique_ptr<int> ptr2 = ptr1;
    std::cout << "The unique pointer into: " << *ptr1 << std::endl;

    // 通过转移所有权的方式初始化
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    std::cout << "The unique pointer into: " << *ptr2 << std::endl;
    std::unique_ptr<int> ptr3 = func();
    std::cout << "The unique pointer into: " << *ptr3 << std::endl;

    // 使用reset方法可以让unique_ptr解除对原始内存的管理，也可以用来初始化一个独占的智能指针
    std::unique_ptr<int> ptr4(new int(12));
    std::unique_ptr<int> ptr5 = std::move(ptr4);

    ptr4.reset();
    ptr5.reset(new int(250));
    // !reset();解除对原始内存的管理
    // std::cout << "The ptr4 unique pointer into: " << *ptr4 << std::endl;
    std::cout << "The ptr5 unique pointer into: " << *ptr5 << std::endl;

    // 如果想要获取独占智能指针管理的原始地址，可以调用get()方法
    std::unique_ptr<int> ptr6(new int(10));
    std::unique_ptr<int> ptr7 = std::move(ptr6);

    ptr7.reset(new int(250));
    std::cout << *ptr7.get() << std::endl; // 得到内存地址中存储的实际数值 250

    // *==============删除器
    using func_ptr = void (*)(int *);
    std::unique_ptr<int, func_ptr> ptr8(new int(10),
                                        [](int *p)
                                        {
                                            delete p;

                                            std::cout << "std::unique deleter\n";
                                        });

    // std::unique_ptr<int, func_ptr> ptr9(new int(10), [&](int *p) { delete p; }); // error
    // ?在lambda表达式没有捕获任何外部变量时,可以直接转换为函数指针,一旦捕获了就无法转换了,
    // ?如果想要让编译器成功通过编译,那么需要使用可调用对象包装器来处理声明的函数指针.

    std::unique_ptr<int, std::function<void(int *)>> ptr10(new int(10),
                                                           [&](int *p)
                                                           {
                                                               delete p;

                                                               std::cout << "std::unique deleter form std::function\n";
                                                           });

    return 0;
}
