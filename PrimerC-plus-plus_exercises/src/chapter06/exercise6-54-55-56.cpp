/* exercise 6-54、6-55、6-56
** 练习6.54: 编写函数的声明，令其接受两个int形参并且返回类型也是int
** 然后声明一个 vector 对象，令其元素是指向该函数的指针。
** solution： 
** 
** 练习6.55: 编写四个函数，分别对两个int值执行加减乘除运算；
** 在上一题创建的 vector 对象职工保存指向这些函数的指针。
** solution: 
**
** 练习6.56: 调用上述 vector 对象中的每个元素并输出结果。
** solution：
*/

#include <iostream>
#include <string>
#include <vector>

// solution 6-54
inline int f(const int, const int);
typedef decltype(f) fp;//fp is just a function type not a function pointer

// solution 6-55
inline int Num_Add(const int n1, const int n2)  { return n1 + n2; }
inline int Num_Sub(const int n1, const int n2)  { return n1 - n2; }
inline int Num_Mul(const int n1, const int n2)  { return n1 * n2; }
inline int Num_Div(const int n1, const int n2)  { return n1 / n2; }

// solution 6-56
std::vector<fp*> v{ Num_Add, Num_Sub, Num_Mul, Num_Div };


int main(int argc, char *argv[])
{
    // solution
    for (auto it = v.cbegin(); it != v.cend(); ++it)
    {
        std::cout << (*it)(2, 2) << std::endl;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-54-55-56.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
