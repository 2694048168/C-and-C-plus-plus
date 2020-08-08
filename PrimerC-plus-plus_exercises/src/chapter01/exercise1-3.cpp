/* exercise 1-3
** 练习1.3：编写程序，在标准输出上打印 Hello，World
*/

// just copy the iostream file
#include <iostream>

// 程序入口 main 主函数
// int 代表程序主函数 main 的返回值类型 
// 返回值，2^8 = 256, [0-255]
int main()
{
    // channel 0 std::cin from Keyboard
    // channle 1 std::cout into Monitor
    // channle 2 std::cerr into Monitor
    std::cout << "Hello, World" << std::endl;
    std::cerr << "Error, " << std::endl;
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-3.cpp
** 3、编译源代码文件并指定标准版本，g++ --version; g++ -std=c++11 -o exercise exercise1-3.cpp
** 4、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 5、将 std::cout 结果重定向到文件，exercise 1> cout_file.txt; Ubuntu使用 ./exercise 1> cout_file.txt
** 6、将 std::ceer 结果重定向到文件，exercise 2> ceer_file.txt; Ubuntu使用 ./exercise 2> ceer_file.txt
** 7、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
