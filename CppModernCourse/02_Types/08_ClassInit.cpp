/**
 * @file 08_ClassInit.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 全功能类的初始化 Fully Featured Classes
 * 与基本类型和POD不同, 全功能类总是被初始化,
 * 一个全功能类的构造函数总是在初始化时被调用, 具体是哪个构造函数被调用取决于初始化时给出的参数.
 * 
 */

class Taxonomist
{
public:
    ~Taxonomist()
    {
        printf("Taxonomist::~Taxonomist is called\n");
    }

    Taxonomist(char x)
    {
        printf("char: %c\n", x);
    }

    Taxonomist(int x)
    {
        printf("int: %d\n", x);
    }

    Taxonomist(float x)
    {
        printf("float: %f\n", x);
    }

    Taxonomist()
    {
        printf("(no argument)\n");
    }
};

// -------------------------------------
int main(int argc, const char **argv)
{
    Taxonomist t1;
    Taxonomist t2{'c'};
    Taxonomist t3{65537};
    Taxonomist t4{6.02e23f};
    Taxonomist t5('g');
    Taxonomist t6 = {'l'};
    Taxonomist t7{};
    Taxonomist t8(); // ! like function call()
                     /**
     * @brief 1. 没有任何大括号或小括号时, 无参数构造函数被调用, 
     * 与POD和基本类型不同, 无论在哪里声明对象, 都可以依赖这种初始化;
     * 2. 使用大括号初始化列表, char、int和 float 构造函数会如预期那样被调用;
     * 3. 可以使用小括号 和等号加大括号的语法, 这些都会调用预期的构造函数;
     * NOTE: 不幸的是, 使用小括号的初始化会导致一些令人惊讶的行为, 它不会给出任何输出.
     * 乍看起来, 最后一个初始化语句像函数声明, 这是因为它就是,
     * 由于一些神秘的语言解析规则, 向编译器声明的是一个尚未定义的函数t8, 
     * 它没有任何参数, 返回一个类型为 Taxonomist 的对象.
     * 
     * ! 注意 函数声明, 在声明中定义函数的修饰符、名称、参数和返回类型, 然后再在函数定义中提供函数体.
     * 这个广为人知的问题被称为"最令人头疼的解析"(most vexing parse),
     * 这也是C++社区在语言中引入大括号初始化语法的一个主要原因.
     *
     * 缩小转换(narrowing conversion)是另一个问题.
     * * 每当遇到隐式缩小转换时, 大括号初始化将产生警告,
     * 这是一个很棒的功能, 可以避免些讨厌的 bug
     * 
     */
    float      a{1};
    float      b{2};

    int narrowed_result(a / b); // !Potentially nasty narrowing conversion
    int result{a / b};          // !Compiler generates warning
    printf("the narrowing conversion result %d\n", narrowed_result);
    printf("the narrowing conversion result %d\n", result);

    /**
     * @brief 初始化类成员
     * 可以使用大括号初始化来初始化类的成员
     * ! 不能使用小括号来初始化成员变量。
     * 
     * *Use braced initializers everywhere
     * C++ 初始化对象的各种方法甚至让有经验的C++程序员都感到困惑,
     * 有一条使初始化变得简单的一般规则: 在任何地方都使用大括号初始化方法.
     * 大括号初始化方法几乎在任何地方都能正常工作而且它们引起的意外也最少,
     * 由于这个原因, 大括号初始化也被称为 "统一初始化" (uniform initialization)
     *
     * !警告 对于C++标准库中的某些类, 可能需要打破在任何地方都使用大括号初始化方法的规则.
     * 
     */
    struct JohanVanDerSmut
    {
        bool gold = true;
        int  year_of_smelting_accident{1970};
        char key_location[8] = {"x-rated"};
    };

    return 0;
}
