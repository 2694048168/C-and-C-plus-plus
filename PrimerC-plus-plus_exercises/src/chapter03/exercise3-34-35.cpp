/* exercise 3-34、3-35
** 练习3.34: 假定 p1 和 p2 指向同一个数组中的元素，则下面程序的功能是什么 ?
** 什么情况下该程序是非法的 ?
** p1 += p2 - p1;
** 跳过 p2 - p1 个元素
** 当 p1 为常量时，非法
**
** 练习3.35: 编写一段程序，利用指针将数组中的元素置为 0
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-35
    int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (auto i : arr)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    for (auto ptr = arr; ptr != arr + 10; ++ptr)
    {
        *ptr = 0;
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
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-34-35.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
