/* exercise 6-52、6-53
** 练习6.52: 已知有如下声明，
** void manip(int, int);
** double dobj;
** 请指出下列调用中每个类型转换的等级
** (a) manip('a', 'z');
** (b) manip(55.4, dobj);
** solution: 
**
** 练习6.53: 说明下列每组声明中的第二条语句会发生说明影响，并指出哪些不合法，如果有
** (a) int cacl(int&, int&);
**     int cacl(const int&, const int&);
** (b) int cacl(char*, char*);
**     int cacl(const char*, const char*);
** (c) int cacl(char*, char*);
**     int cacl(char* const, char* const);
** solution：
*/

#include <iostream>


int main(int argc, char *argv[])
{

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-52-53.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
