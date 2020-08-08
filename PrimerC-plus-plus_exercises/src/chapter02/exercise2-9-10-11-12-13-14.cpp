/* exercise 2-9、2-10、2-11、2-12、2-13、2-14
** 练习2.9:解释下列定义的含义。对于非法的定义，请说明错在何处并将其改正
** (a) std::cin >> int input_value;
** (b) int i={3.14};
** (C) double salary = wage = 9999.99; 
** (d) int i = 3.14;
** solution:
** (a) 非法操作定义，输出流需要一个对象(已经声明过的)，而不能是类型
** (b) 合法操作，声明并初始化变量, 并且类型自动转化为 double
** (c) 非法操作定义，wage 是未声明标识符
** (d) 合法操作，声明并初始化变量，并且类型自动转化为 double
** 
** 练习2.10:下列变量的初值分别是什么?
** std::string global_str;
** int global_int;
** int main ()
** {
**     int local_ int;
**     std::string local_ str;
** }
** solution：
** 全局变量编译器会自动初始化，字符串初始化为空串；int 初始化为 0；
** 局部变量编译器不会自动初始化，随机值，字符串初始化为空串；int 随机初始化；
** 建议所有变量都声明并初始化操作！！！
**
** 练习2.11: 指出下面的语句是声明还是定义:
** (a) extern int ix = 1024;
** (b) int iy;
** (c) extern int iz;
** solution：
** (a) 不能对外部变量进行初始化操作
** (b) 变量声明
** (c) 外部变量声明
**
** 练习2.12: 请指出下面的名字中哪些是非法的?
** (a) int double = 3.14;
** (b) int _ ;
** (C) int catch-22;
** (d) int 1_or_2 = 1;
** (e) double Double = 3.14;
** solution：
** (a) double 是关键字，不能作为标识符
** (b)下划线或者数字不能作为标识符的开头
** (c) 合法标识符 
** (d)下划线或者数字不能作为标识符的开头
** (e) 合法标识符，大小写敏感
**
** 练习2.13: 下面程序中j的值是多少?
** inti=42;
** int main()
** {
**     int i = 100;
**     int j = i;
** }
** solution: i = j = 100
**
** 练习2.14:下面的程序合法吗?如果合法，它将输出什么?
** int i = 100，sum=0;
** for (int i = 0;i != 10; ++i)
**     sum += i;
** std::cout << i << " " << sum <<s td::endl;
** solution: i = 100; sum = 45
**
** 注意变量的作用域以及生命周期！！！
*/

#include <iostream>

// 定义全局变量
int i = 42;

int main()
{
    int i = 100;
    int j = i;
    std::cout << "the value of i : " << i << std::endl;
    std::cout << "the value of j : " << j << std::endl;

    int index = 100, sum = 0;
    for(int index = 0; index != 10; ++index)
        sum += index;
    std::cout << index << " " << sum << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-9-10-11-12-13-14.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
