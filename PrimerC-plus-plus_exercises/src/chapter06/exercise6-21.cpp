/* exercise 6-21
** 练习6.21: 编写函数, 令其接受两个参数，一个是int类型的数，另一个是int指针。
** 函数比较 int 的值和指针指向的值，返回较大的那个。
** 在该函数中指针的类型应该是什么？
** solution：指针类型应该是 const int * ptr
**
*/

#include <iostream>

// solution 6-21
int bigger_integer_(const int integer_one, const int * integer_two)
{
    if (integer_one >= *integer_two)
    {
        return integer_one;
    }
    else
    {
        return *integer_two;
    }
}

int bigger_integer(const int integer_one, const int * integer_two)
{
    return (integer_one >= *integer_two) ? integer_one : *integer_two;
}

int main()
{
    // solution 6-21
    int integer_one = 32, integer_two = 42;
    int *ptr_integer_two = &integer_two;

    std::cout << integer_one << "\t" << integer_two << std::endl;

    std::cout << " the bigger is : " << bigger_integer(integer_one, ptr_integer_two) << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-21.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
