/* exercise 6-33、6-34、6-35
** 练习6.33: 编写一个递归函数，输出vector对象的内容。
**
** 练习6.34: 如果factorial函数的停止条件如下所示，将发生什么情况?
** if (val != 0)
** solution: 递归条件一直执行，不会进入不包含递归的路径，即就是无法结束递归，直至栈空间耗尽
**
** 练习6.35: 在调用factorial 函数时，为什么我们传入的值是val-1而非val--?
** solution：前置 -- 不符合递归算法的求解，后置 -- ，每次都会保留一个副本，如果递归深度太大，则耗内存资源。
**          使用运算，每次进行减法运算，这样不耗内存资源。
**
*/

#include <iostream>
#include <vector>

// solution 6-33
void Recursion_Output(std::vector<int>::const_iterator first, std::vector<int>::const_iterator last)
{
    if (first != last)
    {
        std::cout << *first << " ";
        // Recursion output
        Recursion_Output(++first, last);
    }
}

// solution 6-34 6-35
int factorial(int val)
{
    if (val > 1)
        return factorial(val - 1) * val;
    return 1;
}


int main(int argc, char *argv[])
{
    // solution 6-33
    std::vector<int> ivec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    Recursion_Output(ivec.begin(), ivec.end());
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-33-34-35.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
