/* exercise 1-16
** 练习1.16 : 编写程序，从 cin 中读取一组数据，计算并输出其和
*/

#include <iostream>

int main()
{
    int sum = 0;
    /* EOF End of File 文件结束符
    ** Windows 系统中，首先 Ctrl + Z，然后 Enter 回车，即可向标准输入中输入 EOF
    ** Mac 和 Linux or Unix 系统中，使用 Ctrl + D 即可输入 EOF
    */
    // solution 1
    for (int i = 0; std::cin >> i; sum += i);

    // solution 2
    int value = 0;
    while (std::cin >> value)
    {
        sum += value;
    }    
    
    std::cout << "the result is : " << sum << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-16.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入一组数据如：1 2 3 4 5 6 7 8 9 Ctrl+Z,Enter
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
