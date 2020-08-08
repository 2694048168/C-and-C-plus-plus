/* exercise 1-6
** 练习1.6：解释以下程序片段是否合法，合法输出什么；不合法，原因何在，如何修改
** std::cout << "the sum of " << v1;
**           << "and " << v2;
**           << "is " << v1 + v2 << std::endl;
*/

#include <iostream>

int main()
{
    int value_one = 0, value_two = 0;
    std::cout << "Please enter two numbers:" << std::endl;
    std::cin >> value_one >> value_two;

    // 程序片段不合法    
    // 使用 ；分号，表示一条语句的结束
    // 修改，将第一行和第二行的分号 ；删除即可
    std::cout << "the sum of " << value_one
              << " and " << value_two
              << " is " << value_two + value_one << std::endl;

    std::cout << "the multiplication of " << value_one
              << " and " << value_two
              << " is " << value_two * value_one << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-6.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
