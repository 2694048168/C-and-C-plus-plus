/* exercise 3-6、3-7、3-8
** 练习3.6: 编写一段程序，使用范围 for 语句将字符串内的所有字符用 X 代替
**
** 练习3.7: 就上一题完成的程序而言，如果将循环控制变量的类型设为char将发生什么?
** 先估计一下结果，然后实际编程进行验证
**
** 练习3.8:分别用while循环和传统的for循环重写第一题的程序，你觉得哪种形式更好呢?为什么?
** solution:
** C++ 11 range for is the best more than traditional for and while.
** the new always is more useful than before and the science will go on.
*/

#include <iostream>

int main()
{
    // solutin 3-6
    std::string str = ("this is a string!");
    std::cout << "before the string is : " << str << std::endl;
    // C++11 range for 
    for (auto &i : str)
    {
        i = 'X';
    }
    std::cout << "after  the string is : " << str << std::endl;
    
    // solution 3-7
    std::string str = ("this is a string!");
    std::cout << "before the string is : " << str << std::endl;
    // C++11 range for 
    // auto == char, 这里循环变量 i 本身就是一个 char 类型的引用
    for (char &i : str)
    {
        i = 'X';
    }
    std::cout << "after  the string is : " << str << std::endl;
    
    // solution 3-8
    std::string str_for = ("this is a string!");
    std::cout << "before the string is : " << str_for << std::endl;
    // traditional for 
    for (unsigned i = 0; i <= str_for.size(); ++i)
    {
        str_for[i] = 'X';
    }
    std::cout << "after  the string is : " << str_for << std::endl;

    std::string str_while = ("this is a string!");
    std::cout << "before the string is : " << str_while << std::endl;
    // traditional while
    unsigned j = 0;
    while (j <= str_while.size())
    {
        str_while[j] = 'X';
        ++j;
    }
    std::cout << "after  the string is : " << str_while << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-6-7-8.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入字符，包括回车、空格、制表符，结束输入，键入 EOF 即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
