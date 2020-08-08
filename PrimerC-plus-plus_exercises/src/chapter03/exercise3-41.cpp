/* exercise 3-41
** 练习3.41: 编写一段程序，用整型数组初始化一个 vector 对象
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-41
    int arr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    // 初始化
    std::vector<int> v(std::begin(arr), std::end(arr));

    for (auto i : v)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-41.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
