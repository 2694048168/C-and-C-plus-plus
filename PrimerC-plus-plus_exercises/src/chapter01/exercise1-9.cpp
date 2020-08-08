/* exercise 1-9
** 练习1.9 : 编写程序，使用 while 循环将 50-100 的整数相加
** solution : [50-100]包括首尾两个数，一共 51 个数字，累加次数为 51 次
**          50 + 99 = 149
**          51 + 98 = 149
**          50 / 2 = 25
**          149 * 25 = 3725
**          3725 + 100 = 3825
*/

#include <iostream>

void sum_50_to_100()
{ 
    int i = 0, sum = 0;
    // 循环 51 次后结束 累计和 sum
    while (++i < 52)
    {
        // 起始值为 50 = ++i + 49 = 1 + 49
        sum += (i + 49);
    }   
    std::cout << sum << std::endl;
}

int main()
{
    sum_50_to_100();

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-9.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
