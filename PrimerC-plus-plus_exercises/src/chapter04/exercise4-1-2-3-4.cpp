/* exercise 4-1、4-2、4-3、4-4
** 练习4.1: 表达式 5 + 10 * 20 / 2 的求值结果是多少?
** solution: 先乘除，后加减，左结合 ：105
**
** 练习4.2:根据运算符优先级顺序，在下述表达式的合理位置添加括号，
** 使得添加括号后运算对象的组合顺序与添加括号前一致。
** (a) *vec.begin()
** (b) *vec.begin() + 1
** solution: 函数调用优于解引用，解引用优于四则运算
** (a) (*(vec.begin()))
** (b) ((*(vec.begin())) + 1)
**
** 练习4.3: C++语言没有明确规定大多数二元运算符的求值顺序，给编译器优化留下了余地
** 这种策略实际上是在代码生成效率和程序潜在缺陷之间进行了权衡，你认为这可以接受吗? 请说出你的理由。
** solution：这种权衡取舍是可以接受的
**
** 练习4.4: 在下面的表达式中添加括号，说明其求值的过程及最终结果。
** 编写程序编译该(不加括号的)表达式并输出其结果验证之前的推断。
** 12 / 3 * 4 + 5 * 15 + 24 % 4 / 2
** solution: 乘除取余优于加减，左结合 ： 94
** ((((12 / 3) * 4) + (5 * 15)) + ((24 % 4) / 2))
*/

#include <iostream>

int main()
{
    // solution 4-1
    int a = 5 + 10 * 20 / 2;
    std::cout << a << std::endl;

    // solution 4-4
    int b = 12 / 3 * 4 + 5 * 15 + 24 % 4 / 2;
    std::cout << b << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-1-2-3-4.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
