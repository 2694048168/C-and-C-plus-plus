/* exercise 3-5
** 练习3.5: 编写程序从标准输入中读入多个字符串并将它们连接在一起，输出连接成的大字符串
** 然后修改上述程序，用空格把输入的多个字符串分隔开来
** solution:
*/

#include <iostream>

int main()
{
    // solutin 1
    std::string connect_string;
    std::string buffer;
    while (std::cin >> buffer)
    {
        connect_string += buffer;
    }
    std::cout << "the connection of strings is : " << connect_string << std::endl;
    
    // solution 2
    std::string str;
    std::string buff;
    while (std::cin >> buff)
    {
        str += (str.empty() ? "" : " ") + buff;
    }
    std::cout << "the str is : " << str << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-5.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入字符，包括回车、空格、制表符，结束输入，键入 EOF 即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
