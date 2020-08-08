/* exercise 2-39、2-40
** 练习2.39: 编译下面的程序观察其运行结果，
** 注意，如果忘记写类定义体后面的分号会发生什么情况?记录下相关信息，以后可能会有用。
** struct Foo {  } //注意:没有分号
** int main ()
** {
**     return 0;
** } 
**
** 练习2.40: 根据自己的理解写出Sales_data 类，最好与书中的例子有所区别
**
** 练习2.41: 根据自己写 Sales_data 类，完成以前需要类的练习重写
** 
*/

#include <iostream>

// solution 2-39
// error: expected ';' after struct definition
// struct Foo {}
struct Foo
{
};

// solutin 2-40
// Shift + Alt + F 格式化代码
struct Sales_data
{
    std::string book_isbn;
    unsigned sale_num = 0;
    double revenue = 0.0;
};

int main()
{

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-39-40.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
