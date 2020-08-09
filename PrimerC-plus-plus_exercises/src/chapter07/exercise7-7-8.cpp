/* exercise 7-7、7-8
** 练习7.7: 使用这些新函数重写练习中的交易处理程序
** solution: 
**
** 练习7.8: 为什么 read 函数将 参数定义为普通引用，而print将参数定义为常量引用？
** solution：read 函数中操作会改变流中的数据；而print函数操作中不需要。
*/

#include <iostream>
#include "exercise7-6.hpp"

int main(int argc, char **argv)
{
    Sales_data total;
    if (read(std::cin, total))
    {
        Sales_data trans;
        while (read(std::cin, trans)) {
            if (total.isbn() == trans.isbn())
                total.combine(trans);
            else {
                print(std::cout, total) << std::endl;
                total = trans;
            }
        }
        print(std::cout, total) << std::endl;
    }
    else
    {
        std::cerr << "No data?!" << std::endl;
        return -1;
    }
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-7-8.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
