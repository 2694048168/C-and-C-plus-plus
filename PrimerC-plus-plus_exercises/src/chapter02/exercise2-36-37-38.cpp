/* exercise 2-36、2-37、2-38
** 练习2.36: 关于下面的代码，请指出每一个变量的类型以及程序结束时它们各自的值。
** int a = 3, b = 4;
** decltype(a) c = a;
** decltype( (b)) d = a;
** ++c;
** ++d;
**
* 练习2.37: 赋值是会产生引用的一类典型表达式，引用的类型就是左值的类型。
** 也就是说，如果i是int，则表达式i=x的类型是int&。
** 根据这一特点，请指出下面的代码中每一个变量的类型和值。
** int a = 3,b = 4;
** decltype(a) c = a;
** decltype(a = b) d = a;
**
** 练习2.38:说明由decltype指定类型和由auto指定类型有何区别。
** 请举出一个例子，decltype指定的类型与auto指定的类型一样;
** 再举一个例子，decltype指定的类型与auto指定的类型不一样。
**
** summary: auto and decltype is the new features in C++11
** auto 类型说明符，编译器分析表达式的类型，表达式的值赋给变量，
**      该变量的类型使用auto，并且必须初始化
** decltype 类型指示符，编译器分析表达式的类型，并定义变量，
**      同时不想用该表达式的值进行初始化变量
*/

#include <iostream>

int main()
{
    // solution 2-36
    // VSCode 中，将鼠标放在变量上，编译器会自动识别到其类型
    // b = 4 c = 4 d = a = 4
    int a = 3, b = 4;
    decltype(a) c = a;
    decltype((b)) d = a;
    ++c;
    ++d;
    std::cout << a << "\t" << b << "\t" << c << "\t" <<  d << "\t" << std::endl; 

    // solutin 2-37
    // int a = 3,b = 4;
    // decltype(a) c = a;
    decltype(a = b) d2 = a;
    std::cout << a << "\t" << b << "\t" <<  d2 << "\t" << std::endl; 

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-36-37-38.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
