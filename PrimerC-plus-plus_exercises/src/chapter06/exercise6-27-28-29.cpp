/* exercise 6-27、6-28、6-29
** 练习6.27: 编写一个函数，它的参数是initializer_ list<int> 类型的对象，
**         函数的功能是计算列表中所有元素的和
**
** 练习6.28: 在error_msg 函数的第二个版本中包含ErrCode类型的参数，其中循环内的elem是什么类型?
** solution: elem的类型是通过编译器推断出来的auto，
**          其类型是 const string & 类型。使用引用是为了避免拷贝过长的 string 类型字符串。
**
** 练习6.29: 在范围for循环中使用initializer_ list 对象时，应该将循环控制变量声明成引用类型吗?为什么?
** solution: 引用的优势是在于可以直接使用引用，从而达到操作引用绑定的对象，以及为了避免拷贝时过于复杂。
**          由于initializer_list对象中列表的元素都是const对象，不能修改，
**         所以没必要使用引用类型的控制变量，但是若是string类型或者其他容器类型的对象，
**         执行拷贝操作，有时候会拷贝过长的string对象，所以使用引用是为了避免拷贝.
*/

#include <iostream>

// solution 6-27
// 可变参数列表
int parma_sum(std::initializer_list<int> const & integer_value)
{
    int sum = 0;
    for (auto i : integer_value)
    {
        sum += i;
    }

    return sum;
}

int main(int argc, char *argv[])
{
    auto integer_value = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    std::cout << parma_sum(integer_value) << std::endl;

    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-27-28-29.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
