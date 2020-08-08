/* exercise 1-2
** 练习1.2：改写程序，返回 -1，
## 返回值 -1 通常被当做程序错误的标识
** 重新编译并运行程序，观察系统如何处理 main 返回的错误标识
*/

// 程序入口 main 
int main()
{
    // 返回值，2^8 = 256, [0-255]
    // 返回值数字，代表程序结束时的不同意思
    // 0 代表正常退出结束，-1 代表有异常退出结束，code 0，code 1
    return -1;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-2.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
