/* exercise 1-20
** 练习1.20 : 拷贝 Sales_item.hpp 到工作目录，
** 编写程序，读取一组书籍的销售记录，并将每条记录打印到标准输出上
*/

#include <iostream>
#include "Sales_item.hpp"

int main()
{
    Sales_item item;
    // solution 1
    // EOF
    for (item; std::cin >> item; std::cout << item << std::endl);

    // solution 2
    // EOF
    while (std::cin >> item)
    {
         std::cout << item << std::endl;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-20.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入例子 ：0-201-78345-x 3 20.00
**              0-202-78345-x 2 53.00
**              0-203-78345-x 5 45.00
**              0-204-78345-x 6 25.00
**              0-205-78345-x 1 30.00
**              Ctrl + Z, Enter
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
