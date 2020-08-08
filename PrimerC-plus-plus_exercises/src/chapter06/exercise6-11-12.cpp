/* exercise 6-11、6-12
** 练习6.11: 编写函数并验证 reset 函数，使其作用于引用类型的参数
**
** 练习6.12：改写 6.10 练习程序，使用引用而非指针交换两个整数的值。
** 哪一个方法更易于使用呢？ 为什么？
*/

#include <iostream>

// solution 6-11
void reset(int &i)
{
    i = 0;
}

// solution 6-12
void swap(int &integer_one, int &integer_two)
{
    int temp;
    temp = integer_one;
    integer_one = integer_two;
    integer_two = temp;
}

int main()
{
    // solution 6-11
    int integer = 99;
    std::cout << "before: " << integer << std::endl;
    reset(integer);
    std::cout << "after: " << integer << std::endl;


    // solution 6-12
    int integer_one = 32, integer_two = 42;
    std::cout << "before: " << integer_one << "\t" << integer_two << std::endl;
    swap(integer_one, integer_two);
    std::cout << "after: " << integer_one << "\t" << integer_two << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-11-12.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
