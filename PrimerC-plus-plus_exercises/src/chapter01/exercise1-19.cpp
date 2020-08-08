/* exercise 1-19
** 练习1.19 : 编写程序，提示用户输入两个整数，
** 然后输出打印两个整数所指定范围内的所有整数
** 并且能够识别到用户输入的第一个值比第二个值小的情况
** solution: 
** step0, 比较 integer_one 和 integer_two 的大小
** step1, integer_one > integer_two, and then integer_one-- 到 integer_two
** step2, or then integer_one++ 到 integer_two
*/

#include <iostream>

int main()
{
    std::cout << "Please enter two integer : ";
    int integer_one = 0, integer_two = 0;
    std::cin >> integer_one >> integer_two;

    // solution
    if (integer_one > integer_two)
    {
        for (integer_one; integer_one >= integer_two; --integer_one)
        {
            std::cout << integer_one << std::endl;
        }   
    }
    else
    {
        for (integer_one; integer_one <= integer_two; ++integer_one)
        {
            std::cout << integer_one << std::endl;
        }        
    }   

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-19.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入 21 42
**        42 21
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
