/* exercise 6-10
** 练习6.10: 编写函数, 使用指针形参交换两个整数的值
** 在代码中调用该函数并输出交换后的结果，验证函数正确性
**
*/

#include <iostream>

// solution 6-10
void swap(int* integer_one, int* integer_two)
{
    int temp;
    temp = *integer_one;
    *integer_one = *integer_two;
    *integer_two = temp;
}

int main()
{
    // solution 6-10
    int integer_one = 32, integer_two = 42;
    int *ptr_integer_one = &integer_one, *ptr_integer_two = &integer_two;


    std::cout << "before: " << integer_one << "\t" << integer_two << std::endl;

    swap(ptr_integer_one, ptr_integer_two);

    std::cout << "after: " << integer_one << "\t" << integer_two << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-10.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
