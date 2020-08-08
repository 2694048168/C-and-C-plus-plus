/* exercise 6-18、6-19、6-20
** 练习6.18: 为下面的函数编写函数声明，从给定的名字中推测函数具备的功能。
** (a) 名为compare的函数，返回布尔值，两个参数都是matrix类的引用。
** (b) 名为change_val的函数，返回vector<int>的迭代器，有两个参数:一个是int，另一个是vector<int>的迭代器。
** solution：
**
** 练习6.19: 假定有如下声明，判断哪个调用合法、哪个调用不合法。对于不合法的函数调用，说明原因。
** double calc (double);
** int count (const string &， char);
** int sum(vector<int>::iterator, vector<int>::iterator, int);
** vector<int> vec(10);
** (a) calc(23.4，55.1);
** (b) count ("abcda", 'a');
** (c) calc(66);
** (d) sum (vec.begin(), vec.end()， 3.8);
** solution：
**
** 练习6.20: 引用形参什么时候应该是常量引用?如果形参应该是常量引用，
** 而我们将其设为了普通引用，会发生什么情况?
** solution：函数不会改变的形参定义为常量引用，是很有必要的操作。
** 如果定义为普通引用，误导调用者，函数可以修改其实参的值，调用时易出现错误，而且编译也会出现错误。
*/

#include <iostream>
#include <vector>

// solution 6-18
bool compare(matrix &, matrix &);
std::vector<int> change_val(int, std::vector<int>);

int main()
{
    // solution 6-18

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-18-19-20.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
