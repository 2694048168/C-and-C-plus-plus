/* exercise 5-4、5-5、5-6
** 练习5.4: 说明下列例子的含义，如果存在问题，试着修改它
** (a) while (string::iterator iter != s.end()) {}
** (b) while (bool status = find(word) {}
**        if (!status) {}
** solution：
** (a) while (string::iterator iter != s.end() - 1){}
** (b) 逻辑没有问题 
**
** 练习5.5: 写一段程序，使用 if else 语句实现数字成绩转换字母成绩的要求
** 
** 练习5.6: 改写以上题，使用条件运算符实现 代替 if else 语句
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 5-5
    std::vector<std::string> grade = { "F", "D", "C", "B", "A", "A++" };
    for (int score; std::cin >> score;)
    {
        std::string letter;
        if (score < 60)
        {
            letter = grade[0];
            std::cout << letter << std::endl;
        }
        else
        {
            letter = grade[(score - 50) / 10];
            if (score != 100)
                letter += score % 10 > 7 ? "+" : score % 10 < 3 ? "-" : "";
            std::cout << letter << std::endl;
        }
    }

    // solution 5-6
    int score_grade = 0;
    while (std::cin >> score_grade)
    {
        std::string letter_grade = score_grade < 60 ? grade[0] : grade[(score_grade - 50) / 10];
        letter_grade += (score_grade == 100 || score_grade < 60) ? "" : (score_grade % 10 > 7) ? "+" : (score_grade % 10 < 3) ? "-" : "";
        std::cout << letter_grade << std::endl;
    }


    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-4-5-6.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
