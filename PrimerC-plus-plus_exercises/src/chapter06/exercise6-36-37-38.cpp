/* exercise 6-36、6-37、6-38
** 练习6.36: 编写一个函数声明，使其返回数组的引用并且该数组包含10个string对象。
** 不要使用尾置返回类型、decltype或者类型别名 
** solution：
** 
** 练习6.37: 为上一题的函数再写三个声明，一个使用类型别名，另一个使用尾置返回类型，
** 最后一个使用decltype关键字，你觉得哪一种形式最好，为什么？
** solution: 类型别名
**
** 练习6.38: 修改 arrPtr 函数，使其返回数组的引用
**
*/

#include <iostream>

// solution 6-36
std::string (&function_1()) [10];

// solution 6-37
// 类型别名 str_arr
typedef std::string str_arr[10];
using str_arr = std::string[10];
str_arr & function_4(std::string str);

// 尾置返回类型
auto function_2(std::string str) -> std::string (&)[10];

// decltype 关键字
std::string arr[10] = {"wei", "li", "yzzcq"};
decltype(arr) & function_3(std::string str);

// solution 6-38
int odd[] = {1, 3, 5, 7, 9};
int even[] = {0, 2, 4, 6, 8};

decltype(odd) & arrPtr(int i)
{
    return (i % 2) ? odd : even;
}


int main(int argc, char *argv[])
{
    // solution 6-38
    std::cout << arrPtr << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-36-37-38.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
