/* exercise 4-5、4-6、4-7、4-8、4-9
** 练习4.5:写出下列表达式的求值结果。
** (a) -30 * 3 + 21 / 5
** (b) -30 + 3 * 21 / 5
** (c) 30 / 3 * 21 % 5
** (d) -30 / 3 * 21 % 4
** solution:
** (a) -86
** (b) -18
** (c) 0
** (d) -2
**
** 练习4.6: 写出一条表达式用于确定一个整数是奇数还是偶数。
** solution: integer / 2 ? "True" : "False"
**
** 练习4.7: 溢出是何含义?写出三条将导致溢出的表达式。
** solution: 溢出就是计算的结果超出了该类型所能表示的范围，溢出的结果是无法预知的。
** short short_value = 32767 + 1;
** int int_value = 2^32 + 1;
** double double_value = 2^32 + 1;
**
** 练习4.8: 说明在逻辑与、逻辑或及相等性运算符中运算对象求值的顺序。
** solution: 逻辑与运算符和逻辑或运算符都是先求左侧运算对象的值再求右侧运算对象的值，
** 当且仅当左侧运算对象无法确定表达式的结果时才会计算右侧运算对象的值。
** 这种策略称为短路求值( short-circuit evaluation)。
** 1、对于逻辑与运算符来说，当且仅当左侧运算对象为真时才对右侧运算对象求值。
** 2、对于逻辑或运算符来说，当且仅当左侧运算对象为假时才对右侧运算对象求值。
** == and != 运算符都是左结合，与 && and || 是一致的都是左结合
**
** 练习4.9: 解释在下面的if语句中条件部分的判断过程。
** const char *cp = "Hello World";
** if (cp && *cp)
** solution：先计算指针 cp 是否为非零值，结果为非零值；
**           再计算指针指向的值 *cp 是否为非零值，结果为非零值；
**           逻辑运算 && ，返回结果为 1，True。
**
*/

#include <iostream>

int main()
{
    // solution 4-5
    int a = -30 * 3 + 21 / 5;
    int b = -30 + 3 * 21 / 5;
    int c = 30 / 3 * 21 % 5;
    int d = -30 / 3 * 21 % 4;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;

    // solution 4-9
    const char *cp = "Hello World";
    std::cout << (cp && *cp) << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-5-6-7-8-9.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
