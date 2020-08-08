/* exercise 6-4、6-5
** 练习6.4: 编写一个与用户交互的函数，要求用户输入一个数字，计算生成该数字的阶乘
** 在main函数中调用该函数
**
** 练习6.5: 编写一个函数输出其实参的绝对值。
**
*/

#include <iostream>

// solution 6-4
int factorial(int val)
{
    if (val == 0 || val == 1)
    {
        return 1;
    }
    else
    {
        return val * factorial(val - 1);
    }
}


// solution 6-5
unsigned Absolute(int value)
{
    return abs(value);
}

int main()
{
    // solution 6-4
    std::cout << "Please input an integer : ";
    unsigned val = 0;
    std::cin >> val;
    std::cout << factorial(val) << std::endl;

    // solution 6-5
    std::cout << "Please input an integer : ";
    unsigned value = 0;
    std::cin >> value;
    std::cout << Absolute(value) << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-4-5.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试程序，输入整数即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
