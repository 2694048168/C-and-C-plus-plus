/* exercise 3-2、3-3
** 练习3.2: 编写一段程序从标准输入中一次读入一整行，然后修改该程序使其一次读入一个词
**
** 练习3.3: 请说明string类的输入运算符和getline函数分别是如何处理空白字符的
** solution: 
** 空白符：空格、制表符、换行符
** >> 输入运算符，遇到 空白符 则会结束读取，并返回 string 对象
** getline 函数，遇到 换行符 才会结束读取，并返回不包含换行符的 string 对象
** 两者在遇到 EOF 文件结束符，即流的状态被检测到为无效，使得条件变为假 False；有效状态流检测到，条件为真 True
**
*/

#include <iostream>

int main()
{
    // solution 3-2
    // get a line from the standard input stream std::cin
    std::string line;
    // EOF
    while (getline(std::cin, line))
    {
        // std::endl flush the buffer 
        std::cout << line << std::endl;
    }

    // get a word from the standard input stream std::cin
    std::string word;
    while (std::cin >> word)
    {
        // std::endl flush the buffer 
        std::cout << word << std::endl;
    }
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-2-3.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入 一个字符串 weili is a pretty boy!，然后回车即可, 结束程序输入 EOF即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
