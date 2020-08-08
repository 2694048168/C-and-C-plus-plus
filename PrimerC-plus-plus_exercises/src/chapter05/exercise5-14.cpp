/* exercise 5-14
** 练习5.14: 编写一段程序,从标准输入中读取若干string对象并查找连续重复出现的单词。
** 所谓连续重复出现的意思是: 一个单词后面紧跟着这个单词本身。
** 要求记录连续重复出现的最大次数以及对应的单词。如果这样的单词存在，输出重复出现的最大次数;
** 如果不存在，输出一-条信息说明任何单词都没有连续出现过。
** 例如，如果输入是  how now now now brown COw cow
** 那么输出应该表明单词  now 连续出现了 3 次。
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 5-14
    std::string string_words, pre_word, max_repeat_word;
    unsigned count_repeat = 0, count_max_repeat = 0;
    std::cout << " Please enter the sequence of string : ";
    // 遍历读入的字符序列
    while (std::cin >> string_words)
    {
        if (string_words == pre_word)
        {
            ++count_repeat;
        }
        else
        {
            count_repeat = 1;
            pre_word = string_words;
        }

        // 统计最大次数单词
        if (count_max_repeat < count_repeat)
        {
            count_max_repeat = count_repeat;
            max_repeat_word = pre_word;
        }

    } // while end
      // 统计遍历后的最终结果
    if (count_max_repeat <= 1)
    {
        std::cout << "no word was repeated" << std::endl;
    }
    else
    {
        std::cout << "the word '" << max_repeat_word << "' occurred " << count_max_repeat << " times" << std::endl;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-14.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入一段文本，how now now now brown COw cow 测试程序
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
