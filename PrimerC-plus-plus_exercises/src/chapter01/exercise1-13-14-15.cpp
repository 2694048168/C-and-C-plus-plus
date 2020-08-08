/* exercise 1-13、1-14、1-15
** 练习1.13 : 使用 for 完成 50-100 累加和
** 练习1.14 : for 和 while 循环优缺点
** 练习1.15 : 编程中不断试错，熟悉编译器对常见错误的而生成的错误信息
*/

#include <iostream>

int main()
{
    // solution 1.13
    int sum = 0;
    for (int i = 50; i <= 100; ++i)
    {
        sum += i;
    }
    std::cout << "the result of 50 to 100 is : " << sum << std::endl;
    
    // solution 1.14
    /* for循环书写简练，对内存较节省（局部变量i再循环结束后自动清除）
    ** while循环，可以对一些不确定循环次数的循环进行较好的控制
    */

    // solution 1.15
    /* 1、语法错误，syntax error, 例如 缺少分号 ;
    ** 2、类型错误，type error，例如 一个常量不能作为左值（左值和右值）
    ** 3、声明错误，declaration，例如 变量需要先声明后使用
    ** 4、业务逻辑错误，这个编译器检查不出来
    */

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-13-14-15.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
