/* exercise 6-13、6-14、6-15
** 练习6.13: 假设T是某种类型的名字，说明以下两个函数声明的区别:
**           一个是voidf(T)，另一个是void f(&T)。
** solution：一个形参类型是 T，另一个形参类型是 T 的引用
**
** 练习6.14: 举一个形参应该是引用类型的例子，再举一个形参不能是引用类型的例子
** solution：如果希望传入的值随之而改变，则是引用类型；否则就使用值传递。
**
** 练习6.15: 说明find_ char 函数中的三个形参为什么是现在的类型，
** 特别说明为什么s是常量引用而occurs是普通引用?
** 为什么s和ooccurs是引用类型而c不是?
** 如果令s是普通引用会发生什么情况?
** 如果令occurs是常量引用会发生什么情况?
**
** solution：如果需要函数返回多个值，将形参使用引用类型可以实现这个功能效果，隐式的返回多个值
** s 是不需要改变的，而设置为常量引用；occcurs 是计数，需要改变并返回，设置为普通引用，
** c 不需要返回并且局部变量就可以了
*/

#include <iostream>

// solution 6-15
/* 返回位置同时返回出现次数
** 返回 s 中 c 第一次出现的位置索引
** 引用形参 occurs 负责统计 c 出现的总次数
*/
std::string::size_type find_char(const std::string &s, char c, std::string::size_type &occurs)
{
    // flag 标志第一个匹配到的
    auto ret = s.size();
    occurs = 0;
    
    for (decltype (ret) i = 0; i != s.size(); ++i)
    {
        if (s[i] == c)
        {
            // 返回第一次匹配的索引位置
            if (ret == s.size())
            {
                ret = i;
            }
            ++occurs;
        }
    }
    return ret;    
}

int main()
{
    // solution 6-15
    std::string s = "hello, world.";
    size_t occurs = 0;

    auto index = find_char(s, 'o', occurs);

    std::cout << index << "\t" << occurs <<std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-13-14-15.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
