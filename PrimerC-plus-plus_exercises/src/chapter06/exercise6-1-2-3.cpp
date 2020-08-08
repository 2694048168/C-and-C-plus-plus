/* exercise 6-1、6-2、6-3
** 练习6.1: 实参和形参的区别是什么?
** solution: 函数声明或者定义的时候，参数列表称之为形参；函数调用时的参数列表称之为为实参；
** 实参是形参的初始化值
**
** 练习6.2: 请指出下列函数哪个有错误，为什么?应该如何修改这些错误呢?
** (a) int f() {
**         string s;
**         //...
**         return s;
**     }
** (b) f2(int i) {  }
** (c) int calc(int v1，int v1) }
** (d) double square (double x) return x * x;
** solution：
** (a) 返回值类型不匹配
** (b) 没有函数返回值类型
** (c) 函数体不完整{
** (d) 需要函数体{}
**
** 练习6.3: 编写你自己的fact函数，上机检查 是否正确。
**
*/

#include <iostream>

// solution 6-3
int fact(int val)
{
    if (val == 0 || val == 1)
    {
        return 1;
    }
    else
    {
        return val * fact(val-1);
    }
}

int main()
{
    // solution
    std::cout << fact(5) << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-1-2-3.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
