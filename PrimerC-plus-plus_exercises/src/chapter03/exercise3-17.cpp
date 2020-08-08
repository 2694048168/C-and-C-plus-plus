/* exercise 3-17
** 练习3.17: 从cin读入一组词并把它们存入一个 vector 对象，
** 然后设法把所有词都改写为大写形式。输出改变后的结果，每个词占一行
**
*/

#include <iostream>
#include <vector>
#include <cctype>

int main()
{
    // solutin 3-17
    std::cout << "Please enter a sequence of words: ";
    std::vector<std::string> words_sequence;
    std::string sequence;
    // 保存读入的一组词
    while (std::cin >> sequence)
    {
        words_sequence.push_back(sequence);
    }

    // 将词改写成为大写形式
    // 遍历 vector 容器
    for (auto &sequence_vector : words_sequence)
    {
        // 遍历字符串，
        for (auto &sequence_str : sequence_vector)
        {
            // 非小写字符转换为小写 cctype 头文件中 tolower()
            // 非大写字符转换为大写 cctype 头文件中 toupper()
            sequence_str = toupper(sequence_str);
        }
    }
    // 一行输出一个单词
    for (decltype(words_sequence.size()) i = 0; i != words_sequence.size(); ++i) 
    {
        std::cout << words_sequence[i] << std::endl;
    }
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-17.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入包含大小写的字符，例如：weili yzzcq JXUFE LiWei ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
