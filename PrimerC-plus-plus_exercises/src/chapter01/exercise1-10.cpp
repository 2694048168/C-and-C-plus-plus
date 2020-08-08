/* exercise 1-10
** 练习1.10 : 处理 ++ 运算符将对象的值自增1之外，还有 -- 运算符将对象的值自减1
** 编写程序，使用递减运算符在循环中依次递减打印 10-0 之间整数
*/

#include <iostream>

int main()
{
    /* step 0 : 初始化循环变量 i
    ** step 1 ：判断循环条件，是否进入循环
    ** step 2 ：进入循环，执行 code
    ** step 3 ：更新迭代循环变量 i
    ** step 4 ：重复以上 1、2、3、4，直到 step 循环条件为 False 时结束for循环
    */
    for (int i = 10; i >= 0; --i)
    {
        std::cout << i << std::endl;
    }
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-10.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
