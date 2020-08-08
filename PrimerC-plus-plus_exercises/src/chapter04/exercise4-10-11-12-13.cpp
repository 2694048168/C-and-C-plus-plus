/* exercise 4-10、4-11、4-12、4-13
** 练习4.10: 为while循环写一个条件，使其从标准输入中读取整数，遇到42时停止。
** solution: 
** int integer = 0;
** while ((integer != 42) && (std::cin >> integer))
**
** 练习4.11: 书写一条表达式用于测试4个值a、b、c、d的关系，确保a大于b、b大于c、c大于d。
** solution:  a > b && b > c && c > d
** solution:  (((a > b) && (b > c)) && (c > d))
**
** 练习4.12: 假设i、j和k是三个整数，说明表达式 i != j<k的含义。
** solution: 计算 j < k 的结果是 1 或者 0；在计算 i 的值是否不等于 1 或者 0；
**  < 优于 ！= 运算符，都是左结合
**
** 练习4.13: 在下述语句中，当赋值完成后i和d的值分别是多少?
** int i; double d;
** (a) d = i = 3.5;
** (b) i = d = 3.5;
**
*/

#include <iostream>

int main()
{
    // solution 4-13
    int i; double d;
    d = i = 3.5;
    i = d = 3.5;

    std::cout << i << std::endl;
    std::cout << d << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-10-11-12-13.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
