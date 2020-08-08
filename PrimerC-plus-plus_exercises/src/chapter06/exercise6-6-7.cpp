/* exercise 6-6、6-7
** 练习6.6: 说明形参、局部变量以及局部静态变量的区别。
** 编写一个函数，同时用到这三种形式。
** solution: 形参，自动对象；
** 局部变量，形参和函数体内部定义的变量都是局部变量 local variable
** 函数体内变量，使用 static 关键字进行定义的变量，局部静态变量
**
** 练习6.7: 编写一个函数，当它第一次被调用时返回0，以后每次被调用返回值加1。
**
*/

#include <iostream>

// solution 6-7
size_t  count_function_call()
{
    static size_t function_call = 0;

    return ++function_call;
}

int main()
{
    // solution 6-7
    std::cout << count_function_call() - 1 << std::endl;

    std::cout << count_function_call() - 1 << std::endl;

    std::cout << count_function_call() - 1 << std::endl;

    std::cout << count_function_call() - 1 << std::endl;

    std::cout << count_function_call() - 1 << std::endl;

    std::cout << count_function_call() - 1 << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-6-7.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试程序，输入整数即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
