/* exercise 1-22
** 练习1.22 : 拷贝 Sales_item.hpp 到工作目录，
** 编写程序，读取多个 ISBN 相同书籍的销售记录，输出其和
*/

#include <iostream>
#include "Sales_item.hpp"

int main()
{
    // 用户输入的总数量
    Sales_item total;
    // 以 EOF 作为结束符
    if (std::cin >> total) 
    {
        // 逐个获取用户输入，进行处理
        Sales_item trans;
        while (std::cin >> trans) 
        {
            // ISBN 相同则累加
            if (total.isbn() == trans.isbn())
            {
                total += trans;
            }
            // 不同则输出，并读取下一个新的ISBN  
            else 
            {
                std::cout << total << std::endl;
                total = trans;
            }
        }
        // 输出总的统计值
        std::cout << total << std::endl;
    }
    // ISBN 不同输出错误提示
    else 
    {
        std::cerr << "No data?!" << std::endl;
        return -1;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-22.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入例子 ：0-201-78345-x 3 20.00
**              0-201-78345-x 2 20.00
**              0-201-78345-x 2 20.00
**              0-201-78345-x 4 20.00
**              0-201-78345-x 1 20.00
**              ctrl + z, enter
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/