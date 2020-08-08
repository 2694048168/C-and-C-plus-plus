/* exercise 6-47、6-48
** 练习6.47: 改写6.3.2节练习使用递归输出 vector 内容的程序，
** 使其有条件的输出与执行过程有关信息。例如，每次调用时输出 vector 对象的大小。
** 分别在打开和关闭调试器的情况下编译并执行这个程序。
** 
** 练习6.48: 说明下面这个循环的含义，它对 assert 的使用合理吗？
** string s;
** while (cin >> s && s != sought) {}
** assert(cin);
**
** solution: 调试信息 assert 调试控制宏 NDEBUG
**
*/

#include <iostream>
#include <vector>

// solution 6-47
#define NDEBUG

void Print_Vector(std::vector<int> &vec)
{
#ifdef NDEBUG
    std::cout << "vector size: " << vec.size() << std::endl;
#endif
    if (!vec.empty())
    {
        auto tmp = vec.back();
        vec.pop_back();
        Print_Vector(vec);
        std::cout << tmp << " ";
    }
}

int main(int argc, char **argv)
{
    std::vector<int> vec{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    Print_Vector(vec);

    std::cout << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-47-48.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
