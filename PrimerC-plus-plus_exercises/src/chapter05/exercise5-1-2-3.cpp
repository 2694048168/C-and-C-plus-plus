/* exercise 5-1、5-2、5-3
** 练习5.1: 什么是空语句？ 什么时候用到空语句？
** solution：null statement，只含有一个单独的分号;
**
** 练习5.2: 什么是块？ 什么时候用到块？
** solution：compound statement，复合语句，使用一个花括号{}包含多条语句，也称之一个块。
** 
** 练习5.3: 使用逗号运算符重写 while循环，使得不在需要块，
** 观察改写之后代码的可读性提高了还是降低了。
** solution: 可读性下降了，代码简洁
*/

#include <iostream>

int main()
{
    // solution 5-3
    int sum = 0, val = 1;
    //只要val的值小于等于10，while循环就会持续执行 
    while (val <= 10) 
    {
        sum += val; // 将sum + val 赋予sum
        ++val;     //将val加1
    }
    std::cout << "Sum of 1 to 10 inclusive is " << sum << std::endl;

    while (val <= 10) sum += val, ++val;
    std::cout << "Sum of 1 to 10 inclusive is " << sum << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-1-2-3.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
