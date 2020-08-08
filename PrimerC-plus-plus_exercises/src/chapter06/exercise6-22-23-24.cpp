/* exercise 6-22、6-23、6-24
** 练习6.22: 编写一个函数，令其交换两个int指针。
**
** 练习6.23: 参考本节介绍的几个print函数，根据理解编写你自己的版本。
** 依次调用每个函数使其输入下面定义的 i 和 j:
** int i = 0，j[2] = {0，1};
** solution: 数组作为形参
**
** 练习6.24: 描述下面这个函数的行为。如果代码中存在问题，请指出并改正。
** void print (const int ia[10])
** {
**     for (size_t i = 0; i != 10; ++i)
**     {
**         cout << ia[i] << endl;
**     }
** }
**
*/

#include <iostream>

// solution 6-22
void swap_pointer(const int *integer_one, const int *integer_two)
{
    auto temp = integer_one;
    integer_one = integer_two;
    integer_two = temp;
}

// solution 6-24
void print(const int ia[10])
{
    for (size_t i = 0; i != 10; ++i)
    {
        std::cout << ia[i] << std::endl;
    }
}

int main()
{
    // solution 6-22
    int integer_one = 32, integer_two = 42;
    int *ptr_integer_one = &integer_one, *ptr_integer_two = &integer_two;

    std::cout << ptr_integer_one << "\t" << ptr_integer_two << std::endl;
    std::cout << *ptr_integer_one << "\t" << *ptr_integer_two << std::endl;

    swap_pointer(ptr_integer_one, ptr_integer_two);

    std::cout << ptr_integer_one << "\t" << ptr_integer_two << std::endl;
    std::cout << *ptr_integer_one << "\t" << *ptr_integer_two << std::endl;

    // solution 6-24
    int ia[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    print(ia);

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-22-23-24.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
