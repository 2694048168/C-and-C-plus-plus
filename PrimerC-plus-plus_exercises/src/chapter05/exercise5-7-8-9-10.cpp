/* exercise 5-7
** 练习5.7: 改正下列代码段中的错误。
** (a) if (ival1 != ival2)
**         ival1 = ival2
**    else ivall = ival2 = 0;
**
** (b) if (ival < minval)
**         minval = ival;
**         occurs = 1;
**
** (C) if (int ival = get_value())
**         cout << "ival=” << ival << endl;
**     if (!ival)
**         cout << "ival = 0\n";
**
** (d) if (ival = 0)
**         ival = get_value();
** 
** solution：
** (a) if (ival1 != ival2)
**         ival1 = ival2;
**    else ivall = ival2 = 0;
**
** (b) if (ival < minval){
**         minval = ival;
**         occurs = 1;}
**
** (C) if (int ival = get_value())
**         cout << "ival=” << ival << endl;
**     else
**         cout << "ival = 0\n";
**
** (d) if (ival == 0)
**         ival = get_value();
**
** 练习5.8:什么是 “悬垂else” ? C++语言是如何处理else子句的?
** solution: else 和那个 if 进行匹配的问题，dangling else
** C++ 规定else与离它最近的尚未匹配的if进行匹配，消除程序的二义性
**
** 练习5-9: 编写程序，使用一系列if语句统计从 cin 读入的文本有多少元音字母。
**
** 练习5.10: 修改程序，使得能够统计大小写形式的元音字母。
**
** 练习5.11: 修改程序，使得统计空格、制表符和换行符数量。
**
** 练习5.12: 修改程序，使得能够统计以下含有两个字符的字符序列的数量：ff、fl、fi
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 5-9
    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0;
    char ch;
    while (std::cin >> ch)
    {
        if (ch == 'a') ++aCnt;
        else if (ch == 'e') ++eCnt;
        else if (ch == 'i') ++iCnt;
        else if (ch == 'o') ++oCnt;
        else if (ch == 'u') ++uCnt;
    }
    std::cout << "Number of vowel a: \t" << aCnt << '\n'
              << "Number of vowel e: \t" << eCnt << '\n'
              << "Number of vowel i: \t" << iCnt << '\n'
              << "Number of vowel o: \t" << oCnt << '\n'
              << "Number of vowel u: \t" << uCnt << std::endl;
    
    // solutin 5-10
    while (std::cin >> ch)
        switch (ch)
        {
            case 'a':
            case 'A':
                ++aCnt;
                break;
            case 'e':
            case 'E':
                ++eCnt;
                break;
            case 'i':
            case 'I':
                ++iCnt;
                break;
            case 'o':
            case 'O':
                ++oCnt;
                break;
            case 'u':
            case 'U':
                ++uCnt;
                break;
        }
    std::cout << "Number of vowel a: \t" << aCnt << '\n'
              << "Number of vowel e: \t" << eCnt << '\n'
              << "Number of vowel i: \t" << iCnt << '\n'
              << "Number of vowel o: \t" << oCnt << '\n'
              << "Number of vowel u: \t" << uCnt << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-7.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入一段文本即可，测试程序
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
