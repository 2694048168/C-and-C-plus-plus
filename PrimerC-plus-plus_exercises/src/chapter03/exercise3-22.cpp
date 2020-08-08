/* exercise 3-22
** 练习3.22: 修改之前那个输出text第一段的程序，
** 首先把text的第一段全都改成大写形式，然后再输出它
**
** 例如，假设用一个名为text的字符串向量存放文本文件中的数据，其中的元素或者是一句话
** 或者是一个用于表示段落分隔的空字符串。如果要输出text中第一段的内容,
** 可以利用迭代器写一个循环令其遍历text，直到遇到空字符串的元素为止:
** 
** //依次输出text 的每一行直至遇到第一个空白行为止
** for (auto it = text. cbegin(); it != text.cend() && !it->empty(); ++it)
**     cout << *it << endl;
**
*/

#include <iostream>
#include <vector>
#include <cctype>

// solution Iterative string for vector
void iter_string(const std::vector<std::string> &svec)
{
    // iteration
    for (auto it = svec.begin(); it != svec.end(); ++it)
    {
        std::cout << *it << (it != svec.end() - 1 ? " " : "");
    }
}

int main()
{
    /* text test example
    "weili, liwei, JXUFE, 
     liwei is a boy.
     li jia da yao fang."
    */

    std::vector<std::string> text;
    for (std::string line; getline(std::cin, line); text.push_back(line));

    for (auto it = text.cbegin(); it != text.cend() && !it->empty(); ++it)
    {
        std::cout << *it << std::endl;
    }

    // solution 3-22
    for (auto &word : text)
    {
        for (auto &str : word)
        {
            str = toupper(str);
        }
    }
    // output the result
    iter_string(text);
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-22.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
