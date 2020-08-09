/* exercise 7-27、7-28、7-29、7-30
** 练习7.27: 给你自己的Screen类添加move、set、和display函数，
** 通过执行下面的代码检验你的类是否正确
**   Screen myScreen(5, 5, 'X');
**   myScreen.move(4, 0).set('#').display(cout);
**   cout << "\n";
**   myScreen.display(cout);
**   cout<< "\n";
**
** 练习7.28: 如果move、set和display函数的返回类型不是Screen&而是Screen，则在上一个练习中将会发生什么？
** solution：
** Screen对象将cursor移动到了20的位置，后续的操作都将与新返回的副本进行。’#'被赋值到了副本中。
** 第一个display展示的是被赋值过有的副本。第二个display展示的是全X。
**
** 练习7.29: 修改你的Screen 类，令move、set和display函数返回Screen并检查程序的运行结果，
** 在上一个练习中你的推测正确吗？
** solution：
**
** 练习7.30: 通过this指针使用成员的做法虽然合法，但是有点多余。讨论显式地使用指针访问成员的优缺点。
** solution：
** 通过this指针访问成员的优点是可以非常明确地指出访问的是对象的成员，
** 并且可以在成员函数中使用与数据成员同名的形参；缺点是显得多余，代码不够简洁。
**
*/

#include "exercise7-27.hpp"

int main(int argc, char **argv)
{
    Screen myScreen(5, 5, 'X');
    myScreen.move(4, 0).set('#').display(std::cout);
    std::cout << "\n";
    myScreen.display(std::cout);
    std::cout << "\n";
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-27-28-29-30.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
