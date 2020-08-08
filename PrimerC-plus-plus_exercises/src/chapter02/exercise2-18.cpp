/* exercise 2-18
** 练习2.18: 编写程序，分别更改指针的值以及指针指向对象的值
*/

#include <iostream>

int main()
{
    int value = 0;
    int *ptr_to_value = &value;
    std::cout << value << "\t" << ptr_to_value << "\t"  << *ptr_to_value << std::endl;

    // 修改指针的值
    int value_new = 10;
    ptr_to_value = &value_new;
    std::cout << value << "\t" << ptr_to_value  << "\t" << *ptr_to_value << "\t" << value_new << std::endl;

    // 修改指针指向对象的值
    *ptr_to_value = 20;

    std::cout << value << "\t" << ptr_to_value  << "\t" << *ptr_to_value << "\t" << value_new << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-18.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
