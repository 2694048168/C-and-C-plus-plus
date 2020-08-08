/* exercise 3-12、3-13
** 练习3.12: 下列 vector 对象的定义有不正确的吗?如果有，请指出来
** 对于正确的，描述其执行结果;对于不正确的，说明其错误的原因。
** (a) vector<vector<int>> ivec;
** (b) vector<string> svec = ivec;
** (C) vector<string> svec(10，"null");
** solution:
** (a) 正确，其执行结果，声明 ivec 变量为 vector 容器，其储存的对象是 vector int 对象
** (b) 不正确，两个容器最终储存的类型不匹配，string 和 int
** (c) 正确，其执行结果，svec 变量为 vector 容器，其储存对象为 string类型，有 10 个 “null”字符串
**
** 练习3.13: 下列的 vector对象各包含多少个元素?这些元素的值分别是多少?
** (a) vector<int> vl;
** (b) vector<int> v2(10);
** (c) vector<int> v3(10, 42);
** (d) vector<int> v4{10};
** (e) vector<int> v5{10, 42};
** (f) vector<string> v6{10};
** (g) vector<string> v7{10，"hi"};
** solution:
** (a) 0 个元素，值未知
** (b) 10 个元素，值全部为 0
** (c) 10 个元素，值全部为 42
** (d) 1 个元素，值为 10
** (e) 2 个元素，值分别为 10，42
** (f) 10 个元素，值全部为空串
** (g) 10 个元素，值全部为 “hi”
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-12
    std::vector<std::string> svec(10, "null");

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-12-13.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入字符，包括回车、空格、制表符，结束输入，键入 EOF 即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
