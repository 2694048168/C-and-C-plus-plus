/**
 * @file 03_precedenceAssociativity.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <cstdio>
#include <iostream>

/**
 * @brief Operator Precedence and Associativity
 * 运算符优先级和结合性,
 * 当表达式中出现多个运算符时, 运算符优先级和运算符结合性可以决定表达式的解析方式.
 * !添加括号可以改变优先级顺序, 括号的优先级高于所有运算符.
 * 
 * ==== 求值顺序 Evaluation Order
 * 求值顺序为表达式中运算符的执行顺序,
 * !一个常见的误解是优先级和求值顺序等效,
 * 其实它们并不相同, 优先级是一种编译时概念, 决定运算符绑定到操作数的顺序;
 * 求值顺序是一种运行时概念, 决定运算符的执行调度.
 * *通常, C++ 没有明确指定操作数的执行顺序, 编译器才可以拥有更多优化的机会.
 * 
 * ==== 在某些特殊情况下,语言直接决定执行顺序:
 * 1. 在内置逻辑与运算 a && b 和内置逻辑或运算 a || b 中, 可保证 a 在 b 之前执行;
 * 2. 三元运算符可保证在 a ? b : c 中, a 在 b 和 c 之前执行;
 * 3. 逗号运算符可保证在 a, b 中, a 在 b 之前执行;
 * 4. 在 new 表达式中, 构造函数参数在调用分配器函数之前执行;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    int  a = 1;
    // 如果不注意，甚至可能出现未定义行为
    // 因为未指定表达式 ++a 和 a 的顺序, 而 ++a + a 的值取决于哪个表达式先执行
    // 所以 b 的值不能很好地定义.
    auto b = ++a + a; // !warning
    printf("the value of a: %d, and the b: %d\n", a, b);

    /**
     * @brief 自定义字面量 User-Defined Literals
     * 如何声明字面量以及直接在程序中使用的常量, 它们可以帮助编译器将嵌入的值转换为所需的类型.
     * 每一个基本类型都有自己的字面量语法, char 字面量用单引号声明（如 'J'）
     * 而 wchar_t 字面量则用 L 前缀声明, 例如 L'J';
     * 也可以使用 F 或 L 后缀指定浮点数的精度;
     *
     * 还可以创建自定义字面量, 与内置字面量相同, 自定义字面量也从语法上为编译器提供类型信息;
     * 尽管几乎不需要声明自定义字面量, 但值得一提的是, 可能会在库中发现它们的踪迹;
     * 标准库的＜chrono＞头文件中广泛使用了字面量, 从而为程序员提供了一种使用时间类型的简洁语法
     * 例如, 用 700ms 表示 700 毫秒.
     * https://en.cppreference.com/w/cpp/chrono/operator%22%22s
     */
    using namespace std::chrono_literals;

    std::chrono::seconds halfmin = 30s;
    std::cout << "Half a minute is " << halfmin.count()
              << " seconds"
                 " ("
              << halfmin
              << ").\n"
                 "A minute and a second is "
              << (1min + 1s).count() << " seconds.\n";

    std::chrono::duration moment = 0.1s;
    std::cout << "A moment is " << moment.count()
              << " seconds"
                 " ("
              << moment
              << ").\n"
                 "And thrice as much is "
              << (moment + 0.2s).count() << " seconds.\n";

    return 0;
}
