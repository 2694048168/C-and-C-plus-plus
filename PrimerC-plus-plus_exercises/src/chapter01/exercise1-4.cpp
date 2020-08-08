/* exercise 1-4
** 练习1.4：编写程序，使用加法运算符 + 使得两个数相加
** 使用乘法运算符 * 使得两个数相乘，并打印运算结果
*/

#include <iostream>

int main()
{
    int number_one = 0;
    int number_two = 0;
    std::cout << "Please type the first number:" << std::endl;
    std::cin >> number_one;
    std::cout << "Please type the second number:" << std::endl;
    std::cin >> number_two;
    std::cout << "the sum of numbers:" << number_one + number_two << std::endl;
    std::cout << "the multiplication of numbers:" << number_one * number_two << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-4.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入两个整数，程序自动计算其和与乘积
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
