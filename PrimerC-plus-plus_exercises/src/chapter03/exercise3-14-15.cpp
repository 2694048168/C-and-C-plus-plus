/* exercise 3-14、3-15
** 练习3.14: 编写程序，用 cin 读取一组整数并将它们储存 vector 对象中
**
** 练习3.15: 改写以上程序，读入的是字符串
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-14
    std::cout << "Please enter a sequence of integer: ";
    std::vector<int> integer_sequence;
    int int_sequence;
    while (std::cin >> int_sequence)
    {
        integer_sequence.push_back(int_sequence);
    }
    std::cout << "solution 3-14 is over!" << std::endl;

    // solution 3-15
    std::cout << "Please enter a sequence of string: ";
    std::vector<int> string_sequence;
    int str_sequence;
    while (std::cin >> str_sequence)
    {
        string_sequence.push_back(str_sequence);
    }
    std::cout << "solution 3-15 is over!" << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-14-15.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入字符，包括回车、空格、制表符，结束输入，键入 EOF 即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
