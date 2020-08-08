/* exercise 6-9
** 练习6.6: 编写 fact 函数的实现源文件 和 在 main 函数中调用的源文件
** 理解编译器如何进行分离式编译的
**
*/

#include <iostream>
#include "chapter6.hpp"

int main()
{
    // solution 6-9
    std::cout << fact(5) << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-9.cpp fact.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
