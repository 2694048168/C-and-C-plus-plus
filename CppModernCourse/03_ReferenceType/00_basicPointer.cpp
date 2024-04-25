/**
 * @file 00_basicPointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief 引用类型(reference type)存储的是对象的内存地址.
 * 这种类型可以用来实现高效的程序, 许多优雅的设计模式都利用了它们;
 * C++中两种引用类型: 指针(pointer)和引用(reference);
 * 涉及知识点: this、 const 和 auto 
 * 
 * 如何找到对象的地址, 及如何将地址分配给指针变量;
 * 如何进行相反的操作, 即所谓的解引用: 根据给定的指针获得在相应地址的对象本身;
 *
 * 数组: 是管理对象集合的最简单的结构, 以及数组与指针的关系的知识;
 * 作为底层的结构和概念, 数组和指针使用起来是相对危险的,会出现什么问题?
 * 两种特殊的指针:  void 指针和 std::byte 指针, 这些非常有用的类型也有一些特殊的行为,
 * 如何用 nullptr 表示空指针, 以及如何在布尔表达式中使用指针并判断它们是否为空指针
 * 
 */
// -----------------------------------
#include <cstdio>

int main(int argc, const char **argv)
{
    /** 
     * @brief 指针是用来引用内存地址的基本机制,
     *  指针将对象交互所需的两部分信息: 对象的地址和类型 进行编码.
     * 可以通过在目标类型上添加星号(*)来声明指针的类型.
     * 指针的格式指定符是 %p;
     * 指针在大多数C程序中起着关键作用, 但是它们是底层对象,
     * C++提供了更高级、更有效的结构, 避免了直接处理内存地址.
     * 
     */
    int *ptr = nullptr;
    int  num = 42;
    ptr      = &num;
    printf("the value of pointer: %p\n", ptr);
    printf("the pointer point to value: %d\n", *ptr);
    printf("the pointer point to value: %d\n", num);

    /**
     * @brief ====== 寻址变量
     * 可以通过在变量前面加上地址运算符(&)来获得变量的地址,
     * 使用该运算符还可以初始化指针, 使它"指向"相应的变量, 这样的需求在操作系统编程中经常出现.
     * 
     * gettysburg_address 的值每次都不同, 这种变化是由地址空间随机化(这是一种安全特性)造成的,
     * 它打乱了重要内存区域的基地址, 以防止被恶意利用。
     * !地址空间随机化(address space layout randomization)可以防止恶意利用,
     * ?当黑客在程序中发现可利用的机会时, 有时会在用户提供的输入中插入恶意的数据,
     * ?为了防止黑客获得这种机会而设计的第一个安全特性是使所有数据不可执行, 
     * ?如果计算机试图将数据作为代码执行, 这个安全设计就会触发异常, 终止程序.
     *
     * 一些非常狡猾的黑客通过精心制作包含所谓的return-oriented program(面向返回地址的程序)的漏洞,
     * 想出了如何以全新的方式来利用可执行代码的指令. 
     * 这些漏洞可以调用操作系统API将恶意数据标记为可执行的,从而击败了前面提到的数据不可执行的安全措施.
     * *地址空间随机化通过随机化内存地址来应对面向返回地址的攻击程序,
     * *使得利用现有代码变得很困难, 因为攻击者不知道它们在内存中的位置.
     * 
     */
    int gettysburg{};
    printf("gettysburg: %d\n", gettysburg);
    int *gettysburg_address = &gettysburg;
    printf("&gettysburg: %p\n", gettysburg_address);
    /**
     * @note gettysburg address 在x86架构下包含8个十六进制数字(4个字节),
     * 在x64架构下包含16个十六进制数字(8个字节).
     * 因为在现代桌面系统中, 指针大小与CPU的通用寄存器相同;
     * x86架构拥有32位(4字节)通用寄存器, 而x64架构则是64位(8字节)通用寄存器.
     * 
     */
    printf("the size of pointer: %lld", sizeof(gettysburg_address));

    /**
     * @brief ===== 指针解引用
     * 解引用运算符(*)是一个一元运算符, 它可以访问指针所指的对象, 这是地址运算符的逆运算.
     * 给定一个地址, 可以获得驻留在该地址的对象.
     * 许多操作系统的API会返回指针, 如果想访问被引用的对象, 就需要使用解引用运算符.
     * NOTE: 请记住, 在指针指向对象的类型后面加上星号即可声明指针;
     *       但是, 要解引用, 需要在指针前面加上解引用运算符(即星号)
     * 
     */
    int  gettysburg_{};
    int *gettysburg_address_ = &gettysburg_;
    printf("Value at gettysburg_address_: %d\n", *gettysburg_address_);
    printf("Gettysburg Address: %p\n", gettysburg_address_);
    *gettysburg_address_ = 17325;
    printf("Value at gettysburg_address_: %d\n", *gettysburg_address_);
    printf("Gettysburg Address: %p\n", gettysburg_address_);

    /**
     * @brief ====== 成员指针运算符
     * 成员指针运算符或箭头运算符(->)同时执行两个操作:
     * 1. 对指针解引用;
     * 2. 访问被指向的对象的成员.
     * 当处理指向类的指针时, 可以使用该运算符来减少所谓的符号摩擦, 即程序员在代码中表达意图的阻力.
     * 通常需要在各种设计模式中处理指向类的指针, 将指向类的指针作为函数参数传递.
     * 如果接收函数需要与类的成员交互, 那么就可以使用成员指针运算符来做这件事.
     *
     * 可以使用解引用运算符(*)和成员运算符取得同样的结果,虽然这相当于调用成员运算符, 
     * 但它的语法比较啰唆，而且与简单的替代方法相比没有任何好处.
     * NOTE: 使用括号来强调操作的顺序, 如果不明确运算符的优先级规则.
     * 
     */
    class ClockOfTheLongNowControl
    {
    public:
        ClockOfTheLongNowControl()
            : m_year(2019)
        {
        }

        ClockOfTheLongNowControl(int year)
            : m_year(year)
        {
        }

        ~ClockOfTheLongNowControl()
        {
            m_year = 0;
            printf("the deconstructor and set year == %d\n", m_year);
        }

        void add_year()
        {
            ++m_year;
        }

        bool set_year(int year)
        {
            if (year < 2019)
                return false;
            m_year = year;
            return true;
        }

        int get_year()
        {
            return m_year;
        }

    private:
        int m_year;
    };

    ClockOfTheLongNowControl  clock;
    ClockOfTheLongNowControl *clock_ptr = &clock;
    clock_ptr->set_year(2025);
    printf("Address of clock: %p\n", clock_ptr);
    printf("Value of clock's year: %d\n", clock_ptr->get_year());
    printf("Value of clock's year: %d\n", (*clock_ptr).get_year());

    return 0;
}
