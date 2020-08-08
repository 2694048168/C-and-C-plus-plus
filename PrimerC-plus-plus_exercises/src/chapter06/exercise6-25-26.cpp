/* exercise 6-25、6-26
** 练习6.25: 编写一个main 函数，令其接受两个实参，
** 把实参的内容连接成为一个string对象并输出出来
** solution: liwei yzzcq
**
** 练习6.26: 编写一个程序，使其接受本节所示的选型，
** 输出传递给main函数的实参的内容
** solution: prog -d -o ofile data0
**
*/

#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::string str;

    for (int i = 1; i != argc; ++i)
    {
        str += std::string(argv[i]) + "\t";
    }

    std::cout << str << std::endl;

    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-25-26.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试 25 ，exercise weili yzzcq     Ubuntu使用 ./exercise weili yzzcq
** 4、测试 26 ，exercise -d -o ofile data0    Ubuntu使用 ./exercise -d -o ofile data0 
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
