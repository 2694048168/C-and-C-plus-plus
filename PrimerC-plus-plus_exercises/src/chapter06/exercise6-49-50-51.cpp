/* exercise 6-49、6-50、6-51
** 练习6.49: 什么是候选函数？什么是可行函数？
** solution： 
** 
** 练习6.50: 已知有第217页对函数f 的声明，对于下面的每一个调用列出可行函数。
** 其中那个函数是最佳匹配？如果调用不合法，是因为可以可以匹配的函数还是调用具有二义性？
** (a) f(2.56, 42) (b) f(42) (c) f(42, 0) (d) f(2.56, 3.14)  
** solution: 
**
** 练习6.51: 编写函数 f 的四个版本，令其输出一条可以区分的消息。
** 验证上一个练习的答案，如果打错了，反复研究本节的内容。
** solution：
*/

#include <iostream>

// solution 6-51
void f()
{
    std::cout << "f()" << std::endl;
}

void f(int)
{
    std::cout << "f(int)" << std::endl;
}

void f(int, int)
{
    std::cout << "f(int, int)" << std::endl;
}

void f(double, double)
{
    std::cout << "f(double, double)" << std::endl;
}

int main(int argc, char *argv[])
{
    // f(2.56, 42);
    // error: 'f' is ambiguous.
    f(42);
    f(42, 0);
    f(2.56, 3.14);

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-49-50-51.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
