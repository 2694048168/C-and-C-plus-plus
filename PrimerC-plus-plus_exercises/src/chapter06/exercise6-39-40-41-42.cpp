/* exercise 6-39、6-40、6-41、6-42
** 练习6.39: 说明在下面的每组声明中第二条声明语句是何含义
** 如果有非法的声明，请指出来
** (a) int calc(int, int);
**     int calc(const int, const int);
** (b) int get();
**     double get();
** (c) int *reset(int *);
**     double *reset(double *);
**
** solution：
** 
** 练习6.40: 下面的哪一个声明是错误的，为什么？
** (a) int ff(int a, int b = 0, int c = 0);
** (b) char *init(int ht = 24, int wd, char backgrnd);
**
** solution: 
**
** 练习6.41: 下面的哪一个调用时非法的，为什么？
** 哪一个调用虽然合法但是显然与程序猿的初衷不符合，为什么？
** char *init(int ht, int wd = 80, char backgrnd = ' ');
** (a) init();
** (b) init(24, 10);
** (c) init(14, '*');
**
** solution
**
** 练习6.42: 给 make_plural 函数的第二个形参赋予默认实参 's'，
** 利用新版本的函数输出单词 success 和 failure 的单数和复数形式。
** 
*/

#include <iostream>

// solution 6-42
std::string make_plural(size_t ctr, const std::string& word, const std::string& ending = "s")
{
    return (ctr > 1) ? word + ending : word;
}

int main(int argc, char *argv[])
{
    // solution 6-42
    std::cout << "singual: " << make_plural(1, "success", "es") << "\t"
              << make_plural(1, "failure") << std::endl;
    
    std::cout << "plural : " << make_plural(2, "success", "es") << "\t"
              << make_plural(2, "failure") << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-39-40-41-42.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
