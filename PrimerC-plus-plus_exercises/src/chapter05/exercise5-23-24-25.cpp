/* exercise 5-23、5-24、5-25
** 练习5.23: 编写程序，从标准输入读取两个整数，输出第一个数除以第二个数的结果
**
** 练习5.24: 修改程序，使得当第二个数是 0 时抛出异常。
** 先不要设定 catch 子句，运行程序并真的为除数输入 0，看看会发生什么?
**
** 练习5.25: 修改上一题的程序，使用 try 语句块去捕获异常。
** catch 子句应该为用户输出一条提示信息，询问其是否输入新数并重新执行 try 语句块的内容。
**
*/

#include <iostream>

// solution 5-23
void Division_Integer_23(const int integer_one, const int integer_two)
{
    std::cout << integer_one / integer_two << std::endl;
}

// solution 5-24
void Division_Integer_24(const int integer_one, const int integer_two)
{
    // 直接抛出异常
    if (integer_two == 0)
    {
        throw std::runtime_error("divisor is 0");
    }
    std::cout << integer_one / integer_two << std::endl;
}

// solution 5-25
void Division_Integer_25(const int integer_one, const int integer_two)
{
    for (; ; )
    {
        // 捕获异常信息
        try 
        {
            // 抛出异常信息
            if (integer_two == 0)
            {
                throw std::runtime_error("divisor is 0");
            }
            std::cout << integer_one / integer_two << std::endl;
        }
        catch (std::runtime_error err) 
        {
            std::cout << err.what() << std::endl;
            break;
        }
    }
}

int main()
{
    // solution
    int integer_one = 0, integer_two  = 0;
    std::cout << " Please input two integer: ";
    std::cin >> integer_one >> integer_two;

    // 5-23
    Division_Integer_23(integer_one, integer_two);

    // 5-24
    Division_Integer_24(integer_one, integer_two);

    // 5-25
    Division_Integer_25(integer_one, integer_two);

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-23-24-25.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、测试  5-22, 1024 10
** 4、测试  5-23, 1024 0
** 4、测试  5-25, 1024 0
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
