/* exercise 3-27、3-28、3-29
** 练习3.27: 假设 txt_size 是一个无参数的函数，它的返回值是int。
** 请回答下列哪个定义是非法的? 为什么?
** unsigned buf_size = 1024;
** (a) int ia[buf_size];
** (b) int ia[4*7-14];
** (c) int ia[txt_size()];
** (d) char st[11] = "fundamental";
** solution:
** (a) 合法
** (b) 合法
** (c) 合法
** (d) 不合法，字符串储存还有一个 \0 的位置
** 
** 练习3.28: 下列数组中元素的值是什么?
** string sa[10];
** int ia[10];
** int main() 
** {
**     string sa2[10];
**     int ia2[10];
** }
** solution：全局变量 sa 和 ia 默认初始化为空串和 0；
**          局部变量字符串默认初始化空串，int整数数组默认初始化未知值
** advice：所有变量声明并进行初始化
**
** 练习3.29: 相比于 vector 来说，数组有哪些缺点，请列举一些。
** solution：数组和vector一样可以存放任意对象，除了引用，即不存在引用的数组，也不存在引用的vector。
** １ 数组的维度必须是常量表达式，即在初始化是必须给出。整个程序的运行过程中也不会改变。
** 2 数组不允许拷贝和赋值，即不能将数组的内容拷贝到其他数组作为其初始值，但是vector可以。
**３ 数组使用的过程，容易产生数组越界，而相对于vector则可以使用较多的机制来控制，例如使用迭代器
** advice ：尽量使用 vector 容器替换数组
** 
*/

#include <iostream>
#include <vector>

// solution 3-28
std::string sa[10];
int ia[10];

void output_array_string(const std::string str[])
{
    for (auto i : sa)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void output_array_integer(const int integer[])
{
    for (auto i : sa)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}


int main()
{
    // solutin 3-28
    std::string sa2[10];
    int ia2[10];

    output_array_string(sa);
    output_array_string(sa2);
    output_array_integer(ia);
    output_array_integer(ia2);
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-27-28-29.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
