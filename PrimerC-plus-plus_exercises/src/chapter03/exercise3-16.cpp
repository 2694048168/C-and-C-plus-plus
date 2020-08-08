/* exercise 3-16
** 练习3.16: 编写程序，把练习 3.13中 vector对象的容量和具体内容输出出来。
** (a) vector<int> vl;
** (b) vector<int> v2(10);
** (c) vector<int> v3(10, 42);
** (d) vector<int> v4{10};
** (e) vector<int> v5{10, 42};
** (f) vector<string> v6{10};
** (g) vector<string> v7{10，"hi"};
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-16
    std::vector<int> v1;
    std::vector<int> v2(10);
    std::vector<int> v3(10, 42);
    std::vector<int> v4{10};
    std::vector<int> v5{10, 42};
    std::vector<std::string> v6{10};
    std::vector<std::string> v7{10, "hi"};
    // 遍历序列容器 vertor 里面的元素
    // v1, v, v3, v4, v5, v6, v7
    for (auto sequence : v1)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v1 size is : " << v1.size() << std::endl;

    for (auto sequence : v2)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v2 size is : " << v2.size() << std::endl;

    for (auto sequence : v3)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v3 size is : " << v3.size() << std::endl;

    for (auto sequence : v4)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v4 size is : " << v4.size() << std::endl;

    for (auto sequence : v5)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v5 size is : " << v5.size() << std::endl;

    for (auto sequence : v6)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v6 size is : " << v6.size() << std::endl;

    for (auto sequence : v7)
    {
        std::cout << sequence << std::endl;
    }
    std::cout << "the v1 size is : " << v7.size() << std::endl;
    

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-16.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
