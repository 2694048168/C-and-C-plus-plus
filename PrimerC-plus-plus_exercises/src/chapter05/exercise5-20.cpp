/* exercise 5-20
** 练习5.20:编写一段程序，从标准输入中读取string对象的序列直到连续出现两个相同的单词或者所有单词都读完为止
** 使用 while 循环一次读取一个单词，当一个单词连续出现两次时使用 break 语句终止循环
** 输出连续重复出现的单词，或者输出一个消息说明没有任何单词是连续重复出现的。
**
*/

#include <iostream>

int main()
{
    // solution 5-20
    std::string str_sequence;
    std::string word;
    std::cout << " Please input a sequence of string: ";

    // 检测 流 的有效性， EOF
    while (std::cin >> str_sequence)
    {
        if (str_sequence == word)
        {
            break;
        }
        else
        {
            word = str_sequence;
        }     
    }

    // 如果 EOF 结束，则说明没有连续重复的
    if (std::cin.eof())
    {
        std::cout << " the sequence of string had no repeated." << std::endl;
    }
    // 否则就是 break 结束的，说明有连续重复的
    else
    {
        std::cout << str_sequence << " occurs twice in the input sequence." << std::endl;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-20.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试  liwei jxufe.com weili liwei   输入EOF即可退出程序
** 4、测试  liwei jxufe.com weili weili   输入回车键即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
