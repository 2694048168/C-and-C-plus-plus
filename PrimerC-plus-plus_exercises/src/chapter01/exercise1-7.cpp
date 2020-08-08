/* exercise 1-7
** 练习1.7：编写一个包含不正确的嵌套注释的程序
** 观察编译器返回的错误信息
*/

#include <iostream>

int main()
{
    // 这是行注释    
    /* 这是快注释，多行注释
    */
    /* 两种注释方式随机组合，2 * 2 = 4
    ** 行注释 + 行注释 嵌套，这是可以的
    ** 行注释 + 快注释 嵌套，这是可以的
    ** 块注释 + 行注释 嵌套，这是可以的
    ** 块注释 + 块注释 嵌套，这是不可以的
    */
    // 行注释  // 行注释
    // 行注释 /* 块注释 */
    /* 块注释  // 行注释 */
    /* 块注释  /* 块注释 */  */

    // error: expected primary-expression before '/' token
    // error: expected primary-expression before 'return'

    // GUN g++ (MinGW.org GCC Build-20200227-1) 9.2.0

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-7.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
