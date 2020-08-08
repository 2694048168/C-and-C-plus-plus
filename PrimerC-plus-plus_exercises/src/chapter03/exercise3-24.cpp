/* exercise 3-24
** 练习3.24: 请使用迭代器重做 3.20 练习
**
** 练习3.20: 读入一组整数并把它们存入一个vector对象,将每对相邻整数的和输出出来。
** 改写你的程序，这次要求先输出第1个和最后1个元素的和，
** 接着输出第2个和倒数第2个元素的和，以此类推
** solution: 高斯求和步骤！！！
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-24
    std::vector<int> sequence_ivector;
    int sequence_integer;
    // 读取 cin 一组整数，保存到 vector 中
    while (std::cin >> sequence_integer)
    {
        sequence_ivector.push_back(sequence_integer);
    }
    // 输出每对相邻整数的和
    std::cout << "The sum of two adjacent integers: " << std::endl;
    for (auto it = sequence_ivector.begin(); it + 1 != sequence_ivector.end(); ++it)
    {
        std::cout << *it + *(it + 1) << " ";
    }
    std::cout << std::endl;

    // 高斯求和方法
    // 奇数，则中间那个数与自己相加即可
    for (auto left_iter = sequence_ivector.begin(), right_iter = sequence_ivector.end() - 1; left_iter <= right_iter; ++left_iter, --right_iter)
    {
        std::cout << *left_iter + *right_iter << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-24.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
