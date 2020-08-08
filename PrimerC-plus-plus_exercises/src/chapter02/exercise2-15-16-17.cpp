/* exercise 2-15、2-16、2-17
** 练习2.15:下面的哪个定义是不合法的?为什么?
** (a) int ival = 1.01;
** (b) int &rval1 = 1.01;
** (c) int &rval2 = ival;
** (d) int &rval3;
** solution:
** (a) 合法，隐式类型转换
** (b) 非法，引用的初始化必须为左值
** (c) 合法，rval2 为 ival 的引用，实质就是一个指针（值是地址）
** (d) 非法，声明一个引用，必须用一个左值进行初始化
** 
** 练习2.16:考查下面的所有赋值然后回答:哪些赋值是不合法的?为什么?
** 哪些赋值是合法的?它们执行了什么样的操作?
** int i = 0，&r1 = i;
** double d = 0，&r2 = d;
** (a) r2 = 3.14159;
** (b) r2 = r1;
** (c) i = r2;
** (d) r1 = d;
** solution：
** 第一行和第二行赋值合法，i 赋值 1；r1 引用 i 的值；d 赋值 0；r2 引用 d 的值
** (a) 合法，将 r2 重新赋值为 3.14159，并且由于 r2 为 d 的引用，故此 d 也被重新赋值 3.14159
** (b) 合法，将 r2 又一个重新赋值为 r1的值，也就是 i 的值 0，现在由于 引用 机制，r2=d=r1=i=0
** (c) 合法，
** (d) 合法，
** 引用和指针有所不同，但是引用的本质就是指针！！！
** Numpy 中的传播机制 broadcast 
**
** 练习2.17:执行下面的代码段将输出什么结果?
** int i, &ri = i;
** i = 5; ri = 10;
** std::cout << i << " " << ri << std::endl;
**
*/

#include <iostream>

int main()
{
    // solution 2-16
    int i = 0, &r1 = i;
    std::cout << i << " " << r1 << std::endl;
    double d = 0, &r2 = d;
    std::cout << d << " " << r2 << std::endl;
    r2 = 3.14159;
    std::cout << r2 << " " << d << std::endl;
    r2 = r1;
    std::cout << r2 << " " << r1 << std::endl;
    i = r2;
    std::cout << i << std::endl;
    r1 = d;
    std::cout << d << " " << r1 << std::endl;

    // solution 2-17
    int index, &rindex = index;
    index = 5; 
    rindex = 10;
    std::cout << index << " " << rindex << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-15-16-17.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
