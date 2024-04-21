/**
 * @file 06_userDefinedTypes.cpp
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
  * @brief 用户自定义类型 User-Defined Types
  *  用户自定义类型(User-Defined Type)是用户可以定义的类型:
  * 1. 枚举类型: 最简单的用户自定义类型, 枚举类型可以取的值被限制在一组可能的值中,
  *            枚举类型是对分类概念进行建模的最佳选择.
  * 2. 类: 功能更全面的类型, 可以灵活地结合数据和函数, 
  *       只包含数据的类被称为普通数据类(Plain-old-Data, POD).
  * 3. 联合体: 浓缩的用户自定义类型, 所有成员共享同一个内存位置,
  *           联合体本身很危险, 容易被滥用.
  * 
  */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief ----- 枚举类型
     * 使用关键字 enum class 来声明枚举类型, 关键字后面是类型名称和它可以取的值的列表,
     *  这些值是任意的字母-数字字符串, 代表任意想代表的类别, 在实现内部, 这些值只是整数,
     * 但它们允许使用程序员定义的类型而不是可能代表任何东西的整数来编写更安全、更有表现力的代码.
     * 要将枚举变量初始化为一个值, 使用类型的名称后跟两个冒号::和所需的值即可实现.
     * 
     * NOTE: 从技术上讲, 枚举类是两种枚举类型中的一种: 它被称为作用域枚举,
     * 为了与C语言兼容C++也支持非作用域枚举类型, 它是用 enum 而非 enum class 声明的.
     * 主要的区别是, 作用域枚举需要在值前面加上枚举类型和∷, 而非作用域枚举则不需要.
     * 非作用域枚举类比作用域枚举类使用起来更不安全, 所以除非绝对必要, 否则请不要使用.
     * 
     */
    enum class Race : int
    {
        Dinan = 0,
        Teklan,
        Ivyn,
        Moiran,
        Camite,
        Julian,
        Aidan,
        NUM_RACE
    };
    printf("the value of enum Race::Dinan %d\n", Race::Dinan);
    printf("the value of enum Race::Teklan %d\n", Race::Teklan);
    printf("the value of enum Race::Ivyn %d\n", Race::Ivyn);
    printf("the value of enum Race::Moiran %d\n", Race::Moiran);
    printf("the value of enum Race::Camite %d\n", Race::Camite);
    printf("the value of enum Race::Julian %d\n", Race::Julian);
    printf("the value of enum Race::Aidan %d\n", Race::Aidan);
    printf("the value of enum Race::NUM_RACE %d\n", Race::NUM_RACE);

    printf("========= Switch Statements =========\n");
    /**
     * @brief switch语句
     * switch 语句根据 condition(条件)值将控制权转移到几个语句中的一个,
     * condition 值可以是整数或枚举类型的, switch 语句提供了条件性分支,
     * 当 switch 语句执行时, 控制权将转移到符合条件的情况(case语句),
     * 如果没有符合条件表达式的情况, 则转移到默认情况
     * 每个 case 关键字都表示一种情况而 default 关键字表示默认情况.
     * 
     * NOTE: 执行过程将持续到 switch 语句结束或 break 关键字.
     *  注意 每个 case 的大括号可有可无, 但强烈推荐使用.
     */
    // default 语句是一个安全功能,
    // 如果有人在枚举类中添加了新的 race 值, 那么在运行时将检测到这个未知的 race
    Race condition = Race::Dinan;
    switch (condition)
    {
    case Race::Dinan:
    {
        printf("You word hard.");
    }
    break;
    case Race::Teklan:
    {
        printf("You are very strong.");
    }
    break;
    case Race::Ivyn:
    {
        printf("You are a great leader.");
    }
    break;
    case Race::Moiran:
    {
        printf("My, how versatile you are!");
    }
    break;
    case Race::Camite:
    {
        printf("You're incredibly helpful.");
    }
    break;
    case Race::Julian:
    {
        printf("Anything you want!");
    }
    break;
    case Race::Aidan:
    {
        printf("What an enigma.");
    }
    break;
    default:
    {
        printf("Error: unknown race!");
    }
    }

    printf("========= Plain-Old-Data Classes =========\n");

    /**
     * @brief 普通数据类 POD
     * 类是用户自定义的包含数据和函数的类型, 是C++的核心和灵魂.
     * 最简单的类是普通数据类(Plain-Old-Data, POD), POD是简单的容器,
     * 可以把它们看作一种潜在的不同类型的元素的异构数组, 类的每个元素都被称为一个成员(member),
     * 每个POD都以关键词 struct 开头, 后面跟着POD的名称, 再后面要列出成员的类型和名称. 
     * 
     * 声明POD变量就像声明其他变量一样: 通过类型和名称,可以使用点运算符(.)访问变量的成员.
     * NOTE: POD有一些有用的底层特性: 它们与C语言兼容, 
     * 可以使用高效的机器指令来复制或移动它们, 而且它们可以在内存中有效地表示出来.
     * C++保证成员在内存中是按顺序排列的, 尽管有些实现要求成员沿着字的边界对齐, 这取决于CPU寄存器的长度;
     * 一般来说, 应该在POD定义中从大到小排列成员.
     * 
     */
    struct Book
    {
        char name[256];
        int  year;
        int  pages;
        bool hardcover;
    };

    Book neuromancer;
    neuromancer.pages = 271;
    printf("Neuromancer has %d pages\n", neuromancer.pages);

    printf("========= Unions Types =========\n");

    /**
     * @brief 联合体, union 类似于POD, 它把所有的成员放在同一个地方.
     *  可以把联合体看作对内存块的不同看法或解释, 在一些底层情况下是很有用的,
     * 例如, 处理必须在不同架构下保持一致的结构时, 处理与C/C++互操作有关的类型检查问题时, 甚至在包装位域(bitfield)时.。
     * 声明联合体:用 union 关键字代替 struct 即可.
     * 联合体可以被解释成其中的任意成员, 占用的内存与它最大的成员占用的内存一样多.
     * 可以使用点运算符(.)来指定联合体的解释, 从语法上看,这看起来像访问POD的成员, 但它在内部是完全不同的.
     * NOTE: 因为联合体的所有成员都在同一个地方, 所以很容易造成数据损坏.
     * 这就是联合体存在的主要问题: 要靠程序员自己来跟踪哪种解释是合适的,
     *  编译器不会提供帮助, 除了罕见的情况，应该避免使用联合体，当需要多类型变量时应选择的一些更安全的选择.
     */
    union Variant_
    {
        char   string_[10];
        int    integer_;
        double floating_point_;
    };

    Variant_ v_union;
    v_union.integer_ = 42;
    printf("The ultimate answer: %d\n", v_union.integer_);

    v_union.floating_point_ = 2.7182818284;
    printf("Euler's number e: %f\n", v_union.floating_point_);
    printf("A dumpster fire: %d\n", v_union.integer_); // ! ERROR

    return 0;
}
