/* exercise 5-18、5-19
** 练习5.18: 说明下列循环的含义并改正其中的错误。
** (a) do
**         int v1，v2;
**         cout << "Please enter two numbers to sum:" ;
**         if (cin >> v1 >> v2)
**             cout << "Sumis:" << v1 + v2 << endl;
**     while (cin);
**
** (b) do { // ...} 
**     while (int ival = get_ response()) ;
** 
** (c) do { int ival = get_ response ();}
**     while (ival) ;
**
** 练习5.19: 编写一段程序，使用do while 循环重复地执行下述任务:
** 首先提示用户输入两个string对象，然后挑出较短的那个并输出它。
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 5-17
    do
    {
        std::string str_one, str_two;
        std::cout << " Please enter two sequences of string: ";
        std::cin >> str_one >> str_two;

        if (str_one.size() > str_two.size())
        {
            std::cout << str_two << std::endl;
        }
        else if (str_one.size() < str_two.size())
        {
            std::cout << str_one << std::endl;
        }
        else
        {
            std::cout << "the two sequences of string is equal. " << std::endl;
            std::cout << str_one << str_two << std::endl;
        }

    // 检测 流 是否有效
    } while (std::cin);

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-18-19.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试  liwei jxufe.com   输入EOF即可退出程序
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
