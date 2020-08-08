/* exercise 5-11、5-12、5-13
** 练习5.11: 修改程序，使得统计空格、制表符和换行符数量。
**
** 练习5.12: 修改程序，使得能够统计以下含有两个字符的字符序列的数量：ff、fl、fi
**
** 练习5.13: 指出程序的常见错误
** solution：
** (a) 每一个case 后面忘记了 break;
** (b) 每一个case  多条语句需要使用花括号表示成为一个块{}
** (c) 每一个case 有多个值，就写多少个case，不能再一个case写多个值
** (d) case 的值，必须是常量值，不能是变量
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 5-11
    unsigned aCnt = 0, eCnt = 0, iCnt = 0, oCnt = 0, uCnt = 0, spaceCnt = 0, tabCnt = 0, newLineCnt = 0;
    char ch;
    while (std::cin >> std::noskipws >> ch)
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
            case ' ':
                ++spaceCnt;
                break;
            case '\t':
                ++tabCnt;
                break;
            case '\n':
                ++newLineCnt;
                break;
        }
    
    std::cout << "Number of vowel a(A): \t" << aCnt << '\n'
              << "Number of vowel e(E): \t" << eCnt << '\n'
              << "Number of vowel i(I): \t" << iCnt << '\n'
              << "Number of vowel o(O): \t" << oCnt << '\n'
              << "Number of vowel u(U): \t" << uCnt << '\n'
              << "Number of space: \t" << spaceCnt << '\n'
              << "Number of tab char: \t" << tabCnt << '\n'
              << "Number of new line: \t" << newLineCnt << std::endl;
    
    // solutin 5-12
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

    // solution 5-12
    unsigned ffCnt = 0, flCnt = 0, fiCnt = 0;
    char prech = '\0';
    while (std::cin >> std::noskipws >> ch)
    {
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
                if (prech == 'f') ++fiCnt;
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
            case ' ':
                ++spaceCnt;
                break;
            case '\t':
                ++tabCnt;
                break;
            case '\n':
                ++newLineCnt;
                break;
            case 'f':
                if (prech == 'f') ++ffCnt;
                break;
            case 'l':
                if (prech == 'f') ++flCnt;
                break;
        }
        prech = ch;
    }
    
    std::cout << "Number of vowel a(A): \t" << aCnt << '\n'
              << "Number of vowel e(E): \t" << eCnt << '\n'
              << "Number of vowel i(I): \t" << iCnt << '\n'
              << "Number of vowel o(O): \t" << oCnt << '\n'
              << "Number of vowel u(U): \t" << uCnt << '\n'
              << "Number of space: \t" << spaceCnt << '\n'
              << "Number of tab char: \t" << tabCnt << '\n'
              << "Number of new line: \t" << newLineCnt << '\n'
              << "Number of ff: \t" << ffCnt << '\n'
              << "Number of fl: \t" << flCnt << '\n'
              << "Number of fi: \t" << fiCnt << std::endl;

    // solution 5-13

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-11-12-13.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入一段文本，测试程序
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
