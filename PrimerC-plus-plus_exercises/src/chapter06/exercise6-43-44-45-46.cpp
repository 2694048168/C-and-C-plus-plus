/* exercise 6-43、6-44、6-45、6-46
** 练习6.43: 你会把下面哪一个声明和定义放在头文件中，那个放在源文件中，为什么？
** (a) inline bool eq (const BigInt&, const BigInt&){}
** (b) void putValues(int *arr, int size);
** solution： 内联函数和 constexpr 函数声明和定义都放在头文件中
** 
** 练习6.44: 将6.2.2节的 isShorter 函数改写成为内联函数 
** solution: 
**
** 练习6.45: 回顾前面练习中编写的函数，应该是内联函数嘛，
** 如果是，改写为内联函数；如果不是，说明原因？
** solution: 内联机制由于优化规模小、流程直接并且频繁调用的函数。
**
** 练习6.46: 能把 isShorter 函数定义成为 constexpr 函数嘛？
** 如果能，则改写为 constexpr 函数；如果不能，说明原因？
** solution：constexpr 函数是指能用于常量表达式的函数。
*/

#include <iostream>

// solution 6-44
inline bool is_shorter(const std::string &lft, const std::string &rht) 
{
    return lft.size() < rht.size();
}

int main(int argc, char *argv[])
{
    std::cout << is_shorter("yzzcq", "weili") << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-43-44-45-46.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
