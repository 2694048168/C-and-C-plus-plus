/* exercise 4-29、4-30、4-31、4-32、4-33、4-34、4-35、4-36、4-37、4-38
** 练习4.29: 推断下面代码的输出结果并说明理由。
** 实际运行这段程序，结果和你想象的一样吗?如果不一样，为什么?
** int x[10]; int *p = X;
** cout << sizeof (x) /sizeof (*x) << endl;
** cout << sizeof (p) /sizeof (*p) << endl;
** solution: 
**
** 练习4.30: 根据运算符优先级，在下述表达式的适当位置加上括号，使得加上括号之后表达式的含义与原来的含义相同。.
** (a) sizeof x + y
** (b) sizeof p->mem[i]
** (C) sizeof a < b
** (d) sizeof f()
** solution: sizeof 是一个运算符，而不是一个函数
**
** 练习4.31:本节的程序使用了前置版本的递增运算符和递减运算符，
** 解释为什么要用前置版本而不用后置版本。
** 要想使用后置版本的递增递减运算符需要做哪些改动?使用后置版本重写本节的程序。
** 说明前置递增运算符和后置递增运算符的区别。
** solution: 建议:除非必须，否则不用递增递减运算符的后置版本!!!
** 有C语言背景的读者可能对优先使用前置版本递增运算符有所疑问,其实原因非常简单:
** 前置版本的递增运算符避免了不必要的工作，它把值加 1 后直接返回改变了的运算对象。
** 与之相比，后置版本需要将原始值存储下来以便于返回这个未修改的内容。
** 如果不需要修改前的值，那么后置版本的操作就是一种浪费
** 对于整数和指针类型来说，编译器可能对这种额外的工作进行一定的优化;
** 但是对于相对复杂的迭代器类型，这种额外的工作就消耗巨大了。
** 建议养成使用前置版本的习惯，这样不仅不需要担心性能的问题，而且更重要的是写出的代码会更符合编程的初衷。
**
** 练习4.32:解释下面这个循环的含义。
** constexpr int size = 5;
** int ia[size] = {1,2,3,4,5};
** for(int *ptr = ia,ix = 0;
** ix != size && ptr != ia + size;
** ++ix, ++ptr) { // }
** solution：遍历数组，只要遍历个数不等于数组大小同时指针指向没有对数组越界，则继续遍历即可
**
** 练习4.33:根据运算符优先级说明下面这条表达式的含义。
** someValue ? ++x， ++y : --x，--Y
** solution：somevalue 为真则执行 ++x, ++y, 否则执行 --x, --y
**
** 练习4.34:根据本节给出的变量定义,说明在下面的表达式中将发生什么样的类型转换:
** (a) if (fval) 
** (b) dval = fval + ival; 
** (c) dval + ival * cval;
** 需要注意每种运算符遵循的是左结合律还是右结合律。
** solution：类型转换分为显示类型转换和隐式类型转换。
**
** 练习4.35:假设有如下的定义，
** char cval;
** int ival;
** unsigned int ui;
** float fval ;
** double dval ;
** 请回答在下面的表达式中发生了隐式类型转换吗?如果有，指出来。
** (a) cval = 'a’+ 3;
** (b) fval = ui - ival * 1.0;
** (c) dval = ui * fval;
** (d) cval = ival + fval + dval;
** solution：char -> int -> float -> double
**
** 练习4.36: 假设i是int类型，d是double类型，书写表达式 i *= d 使其执行整数类型的乘法而非浮点类型的乘法。
** solution：强制类型转换 i *= int(d)
** 
** 练习4.37:用命名的强制类型转换改写下列旧式的转换语句。
** int i; double d; const string *ps; char *pc; void *pv;
** (a) pv = (void*)ps; 
** (b) i = int(*pc) ;
** (C) pv = &d;
** (d) pc = (char*) pv;
** solution：强制类型转换干扰正常类型的检查。
**
** 练习4.38:说明下面这条表达式的含义。
** double slope = static_cast<double>(j/i) ;
** solution：
**
*/

#include <iostream>

int main()
{
    // solution 4-38

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-29-30-31-32-33-34-35-36-37-38.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
