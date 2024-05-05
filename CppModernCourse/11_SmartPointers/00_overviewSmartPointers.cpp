/**
 * @file 00_overviewSmartPointers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
  * @brief Smart Pointers
  * 
  * 动态对象具有最灵活的生命周期,巨大的灵活性伴随着巨大的责任, 因为必须确保每个动态对象只被销毁一次;
  * 只需考虑异常因素如何影响动态内存管理, 每次发生错误或异常时, 
  * 都需要跟踪已成功进行的分配,并确保以正确的顺序释放它们.
  * 可以使用 RAII 技术 来处理这种乏味的事情, 通过在 RAII 对象的构造函数中获取动态存储空间,
  * 并在析构函数中释放动态存储空间, 就相对不容易出现动态内存泄漏(或重复释放)的情况;
  * 这使能够使用移动语义和复制语义管理动态对象生命周期, 可以自己编写这些 RAII 对象,
  * 但也可以使用一些被称为智能指针的优秀预写实现, 
  * 智能指针是类模板, 其行为类似于指针, 可以为动态对象实现 RAII.
  * 
  * 1. 作用域指针 scoped pointers. (just int Boost, not in Stdlib)
  * 2. 独占指针: unique pointers. (just in Stdlib, not offer in Boost)
  * 3. 共享指针: shared pointers.
  * 4. 弱指针: weak pointers.
  * 5. 侵入式指针: intrusive pointers.
  * 
  * =====智能指针所有权 Smart Pointer Ownership
  * !每个智能指针都有一个所有权模型, 该所有权模型用于指定指针与动态分配的对象的关系.
  * 当智能指针拥有一个对象时, 智能指针的生命周期保证至少与对象的生命周期一样长;
  * 换句话说, 当使用智能指针时, 可以不用担心被指向的对象已被销毁, 并且被指向的对象不会泄漏;
  * 智能指针管理它拥有的对象, 因此不用担心忘记销毁它, 这要归功于 RAII.
  * *在考虑使用哪个智能指针时, 所有权要求决定了你的选择.
  * 
  */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("[INFO]the Scoped Pointer just in Boost, not in Stdlib\n");
    printf("[INFO]the Unique Pointer just in Stdlib, not in Boost\n");
    printf("[INFO]the Shared Pointer recommend use in Stdlib\n");
    printf("[INFO]the Weak Pointer recommend use in Stdlib\n");
    printf("[INFO]the Intrusive Pointer\n");

    return 0;
}
