/* exercise 3-23
** 练习3.23: 编写一段程序，创建一个含有10个整数的vector对象，
** 然后使用迭代器将所有元素的值都变成原来的两倍。
** 输出vector对象的内容，检验程序是否正确
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-23
    std::vector<int> ivec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (auto it = ivec.begin(); it != ivec.end(); ++it)
    {
        *it *= 2;
    }
    // output the result
    for (auto itera : ivec)
    {
        std::cout << itera << " ";
    }
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-23.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
