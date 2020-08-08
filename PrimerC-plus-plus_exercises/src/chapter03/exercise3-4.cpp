/* exercise 3-4
** 练习3.4: 编写程序读入两个字符串，比较其是否相等并输出结果。
** 如果不相等，输出较大的那个字符串。
** 改写上述程序，比较输入的两个字符串是否等长，
** 如果不等长，输出长度较大的那个字符串。
**
*/

#include <iostream>

int main()
{
    // solution 3-4
    // computing the two strings are equal.
    std::string word_one, word_two;
    while (std::cin >> word_one >> word_two)
    {
        if (word_one != word_two && word_one > word_two)  // word_one > word_two
        {
            std::cout << "the bigger string is : " << word_one << std::endl;
        }
        else if (word_one != word_two &&  word_one < word_two)  // word_one < word_two
        {
            std::cout << "the bigger string is : " << word_two << std::endl;
        }
        else  // the two strings are equal.
        {
            std::cout << "the two strings are equal." << std::endl;
        }
    }

    // cpmputing the size of two strings is euqal.
    std::string word_three, word_four;
    while (std::cin >> word_three >> word_four)
    {
        // the size of word_three > the size of word_four
        if (word_three.size() != word_four.size() && word_three.size() > word_four.size())
        {
            std::cout << "the bigger size of string is : " << word_three << std::endl;
            std::cout << "and the size of " << word_three << " string is : " << word_one.size() << std::endl;
        }
        // the size of word_three < the size of word_four
        else if (word_three.size() != word_four.size() &&  word_three.size() < word_four.size())  // word_one < word_two
        {
            std::cout << "the bigger size of string is : " << word_four << std::endl;
            std::cout << "and the size of " << word_four << " string is : " << word_four.size() << std::endl;
        }
        else  // the size of two strings are equal.
        {
            std::cout << "the size of two strings are equal." << std::endl;
        }
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-4.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入 一个字符串 weili，然后回车即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
