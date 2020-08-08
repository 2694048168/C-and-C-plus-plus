/* exercise 2-3、2-4
** 练习2.3: 读程序，说结果
** unsigned u = 10, u2 = 42;
** std::cout << u2 - u << std:endl;
** std::cout << u - u2 << std:endl;
**
** int i = 10, i2 = 42;
** std::cout << i2 - i << std:endl;
** std::cout << i - i2 << std:endl;
** std::cout << i - u << std:endl;
** std::cout << u - i << std:endl;
** solution: 32 error 32 -32 0 0 
**
** 练习2.4: 编写程序，检测自己的估算结果是否正确
** 32
** 4294967264
** 32
** -32
** 0
** 0
*/

#include <iostream>

int main()
{
    unsigned u = 10, u2 = 42;
    std::cout << u2 - u << std::endl;
    std::cout << u - u2 << std::endl;

    int i = 10, i2 = 42;
    std::cout << i2 - i << std::endl;
    std::cout << i - i2 << std::endl;
    std::cout << i - u << std::endl;
    std::cout << u - i << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-3-4.cpp
** 3、编译源代码文件并指定标准版本，g++ --version; g++ -std=c++11 -o exercise exercise2-3-4.cpp
** 4、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
