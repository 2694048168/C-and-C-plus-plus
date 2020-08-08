/* exercise 1-12
** 练习1.12 : 程序中 for 完成了什么功能，sum 最终值多少
** int sum = 0;
** for (int i = -100; i <= 100; ++i)
** {
**     sum += i;
** }
** solution: 
** step0, 完成从 -100 到 100 的累加和
** step1, sum = 0
** step2, sum = 0,并不是因为初值为 0，而是通过计算累和为 0
*/

#include <iostream>

int main()
{
    int sum = 0;
    for (int i = -100; i <= 100; ++i)
    {
        sum += i;
    }
    std::cout << "the result is : " << sum << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-12.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
