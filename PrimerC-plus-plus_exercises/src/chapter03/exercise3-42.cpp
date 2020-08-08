/* exercise 3-42
** 练习3.42: 编写程序，将含有整数元素的 vector 对象拷贝给一个整型数组。
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-42
    std::vector<int> v{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int arr[10];
    // copy
    for (int i = 0; i != v.size(); ++i)
    {
        arr[i] = v[i];
    }

    for (auto i : arr)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-42.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
