/**
 * @file 05_copySemantics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>

/**
 * @brief 复制语义(copy semantics)是"复制的意思"
 * 实践中, 程序员们用这个词来表示对象进行复制的规则: x被复制到y后, 它们是等价且独立的.
 * 在复制后 x==y是 true(等价性), 对x的修改不会引起对y的修改(独立性);
 * 复制行为非常常见, 特别是当通过值将对象传递到函数时.
 * *对于用户自定义POD类型, 按值传递会导致每个成员值被复制到参数中(按成员复制).
 * 
 * ======= 复制指南 Copy Guidelines
 * 当实现复制行为时, 请考虑以下特性:
 * 1. 正确性: 必须确保类不变量得到维护, SimpleString 类演示了默认的复制构造函数可能违反不变量.
 * 2. 独立性: 在复制赋值或复制构造之后, 原始对象和副本对象在修改过程中不应该改变对方的状态.
 * 3. 等价性: 原始对象和副本对象应该是一样的, 等价性的语义取决于上下文.
 * 
 * ====== 默认复制 Default Copy
 * 通常情况下, 编译器会为复制构造函数和复制赋值运算符生成默认的实现,
 * 默认的实现是在类的每个成员上调用复制构造函数或复制赋值运算符.
 * 无论何时, 当类管理资源时, 必须对默认的复制语义极为小心, 它们很可能是错的
 * 针对这样的情况, 最佳实践是要使用 default 关键字显式声明默认的复制赋值运算符和复制构造函数.
 * 有些类根本不能或不应该被复制, 例如管理着文件的类或代表并发编程中的互斥锁的类,
 * 可以使用 delete 关键字阻止编译器生成复制构造函数和复制赋值运算符.
 * *强烈建议大家为拥有资源(如打印机、网络连接或文件)的类明确定义复制赋值运算符和复制构造函数,
 * *如果不需要自定义行为, 则使用 default 或 delete, 可以避免很多讨厌的和难以调试的错误.
 * 
 */
int add_one_to(int x)
{
    x++;
    return x;
}

struct Point
{
    int x, y;
};

Point make_transpose(Point p)
{
    int tmp = p.x;
    p.x     = p.y;
    p.y     = tmp;
    return p;
}

// *Best practice dictates that you explicitly declare that
// *default copy assignment and copy construction are acceptable for
// *such classes using the default keyword.
struct Replicant
{
    Replicant(const Replicant &)            = default;
    Replicant &operator=(const Replicant &) = default;
};

// *suppress the compiler from generating a
// *copy constructor and a copy assignment operator using the delete keyword.
struct Highlander
{
    Highlander(const Highlander &)            = delete;
    Highlander &operator=(const Highlander &) = delete;
};

// --------------------------------------
int main(int argc, const char **argv)
{
    auto original = 1;
    auto result   = add_one_to(original);
    printf("Original: %d; Result: %d\n", original, result);

    auto original_pod = Point{2, 3};
    auto result_pod   = make_transpose(original_pod);
    printf("Original POD: (%d, %d); Result POD: (%d, %d)\n", original_pod.x, original_pod.y, result_pod.x,
           result_pod.y);

    /**
     * @brief 对于基本类型和POD类型, 复制这些类型时是逐个成员复制的,
     * 这意味着每个成员都会被复制到相应的目标中, 实际上是从一个内存地址到另一个内存地址的按位复制.
     * 全功能类(class)需要多考虑一些, 默认复制语义也是按成员逐个复制, 但是这可能是非常危险的.
     * 
     * 考虑一下 SimpleString 类, 如果允许用户对活动的 SimpleString 类进行按成员复制可能会导致灾难性结果,
     * 两个 SimpleString 类将指向同一个 buffer, 由于这两个副本都追加到同一个 buffer, 会互相破坏.
     * 这个结果是有问题的, 但更糟糕的事情发生在 SimpleString 类开始析构的时候,
     * 当其中一个SimpleString 类被销毁时, buffer 将被释放,
     * !当余下的 SimpleString 类试图写入 buffer 时, 未定义行为产生了.
     * !在某个时候, 余下的 SimpleString 类将被析构并再次释放 buffer, 导致通常所说的"重复释放"(double free).
     * NOTE: "释放后使用","重复释放"会导致一些微妙的, 难以诊断的bug, 而这些bug的出现概率非常低.
     * 释放了对象, 它的存储生命周期就结束了, 这块内存现在处于未定义的状态, 
     * 并且如果析构一个已经被析构的对象, 就会有未定义行为产生, 在某些情况下, 这会导致严重的安全漏洞.
     * 可以通过控制复制语义来避免, 可以指定复制构造函数(Copy Constructors)和复制赋值运算符( copy assignment operators)
     * 
     */

    return 0;
}
