/* exercise 3-30、3-31、3-32、3-33
** 练习3.30: 指出下面代码中的索引错误。
** constexpr size_t array_size = 10;
** int ia[array_size];
** for (size_t ix = 1; ix <= array_size; ++ix)
**     ia[ix] = ix;
**
** solution: 数组下标从 0 开始，for 循环数组访问越界了
** 数组越界问题，编译没有任何问题，造成内存泄露，运行出错
** 
** 练习3.31: 编写程序，定义一个含有10个int的数组，令每个元素的值就是其下标值。
**
** 练习3.32: 将上一题刚刚创建的数组拷贝给另外一个数组。
** 利用 vector 重写程序，实现类似的功能。
**
** 练习3.33: 对于104页的程序来说，如果不初始化scores将发生什么?
** solution：数组为进行初始化，则访问一定越界，运行出错
*/

#include <iostream>
#include <vector>

void solution_31()
{
    // solutin 3-31
    std::cout << " solution 3-31 : ";
    constexpr size_t array_size = 10;
    int ia[array_size];
    for (size_t ix = 0; ix < array_size; ++ix)
    {
        ia[ix] = ix;
        std::cout << ia[ix] << " ";
    }
    std::cout << std::endl;
}

void solution_32()
{
    // solution 3-32
    // array
    std::cout << " array copy : ";
    int arr1[10];
    int arr2[10];

    for (int i = 0; i < 10; ++i)
    {
        arr1[i] = i;
    }

    for (int i = 0; i < 10; ++i)
    {
        arr2[i] = arr1[i];
        std::cout << arr2[i] << " ";
    } 
    std::cout << std::endl;

    // vector
    std::cout << " vector copy : ";
    std::vector<int> v(10);

    for (int i = 0; i != 10; ++i)
    {
        v[i] = arr1[i];
    }
    
    for (auto i : v) std::cout << i << " ";
    std::cout << std::endl;
}

int main()
{
    solution_31();
    solution_32();
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-30-31-32-33.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
