/* exercise 3-1
** 练习3.1: 使用 using 声明命名空间
*/

#include <iostream>
// solution 3-1
// using namespace std;
// headfile has no namespace!!!
using std::cout;
using std::cin;
using std::endl;

int main()
{
    std::string name;
    cin >> name;
    cout << " hello, world " << endl;
    cout << " hello, " << name << endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-1.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入 一个字符串 weili，然后回车即可
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
