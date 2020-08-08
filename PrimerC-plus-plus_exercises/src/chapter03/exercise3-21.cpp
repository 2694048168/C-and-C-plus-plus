/* exercise 3-21
** 练习3.21: 请使用迭代器重做 3.13练习
** (a) vector<int> vl;
** (b) vector<int> v2(10);
** (c) vector<int> v3(10, 42);
** (d) vector<int> v4{10};
** (e) vector<int> v5{10, 42};
** (f) vector<string> v6{10};
** (g) vector<string> v7{10，"hi"};
*/

#include <iostream>
#include <vector>

// solution Iterative integer for vector
void iter_integer(const std::vector<int> &ivec)
{
    std::cout << "size: " << ivec.size() << "  content: [ ";
    // iteration
    for (auto it = ivec.begin(); it != ivec.end(); ++it)
    {
        std::cout << *it << (it != ivec.end() - 1 ? "\t" : "");
    }
    std::cout << " ]" << std::endl;
}

// solution Iterative string for vector
void iter_string(const std::vector<std::string> &svec)
{
    std::cout << "size: " << svec.size() << "  content: [ ";
    // iteration
    for (auto it = svec.begin(); it != svec.end(); ++it)
    {
        std::cout << *it << (it != svec.end() - 1 ? "\t" : "");
    }
    std::cout << " ]" << std::endl;
}

int main()
{
    // solution 3-21
    std::vector<int> v1;
    std::vector<int> v2(10);
    std::vector<int> v3(10, 42);
    std::vector<int> v4{10};
    std::vector<int> v5{10, 42};
    std::vector<std::string> v6{10};
    std::vector<std::string> v7{10, "hi"};

    // iteration
    iter_integer(v1);
    iter_integer(v2);
    iter_integer(v3);
    iter_integer(v4);
    iter_integer(v5);
    iter_string(v6);
    iter_string(v7);
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-21.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
