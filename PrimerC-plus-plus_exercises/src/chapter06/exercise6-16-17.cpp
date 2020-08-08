/* exercise 6-16、6-17
** 练习6.16: 下面的这个函数虽然合法，但是不算特别有用。指出它的局限性并设法改善。
** bool is_empty (string& s) { return s.empty(); }
** solution：
**
** 练习6.17: 编写一个函数，判断string对象中是否含有大写字母。
** 编写另一个函数，把string对象全都改成小写形式。
** 在这两个函数中你使用的形参类型相同吗?为什么?
** solution: 形参类型相同，处理的对象以及操作类似
*/

#include <iostream>

// solution 6-17
bool any_capital(std::string const &str)
{
    for (auto ch : str)
        if (isupper(ch)) return true;
    return false;
}

void to_lowercase(std::string &str)
{
    for (auto &ch : str) ch = tolower(ch);
}


int main()
{
    // solution 6-17
    std::string str_sequence("Hello World!");
    std::cout << str_sequence << std::endl;

    std::cout << any_capital(str_sequence) << std::endl;

    to_lowercase(str_sequence);
    std::cout << str_sequence << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-16-17.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
