/* exercise 1-25
** 练习1.25 : 拷贝 Sales_item.hpp 到工作目录，
** 编译并运行程序即可
*/

#include <iostream>
#include "Sales_item.hpp"

int main()
{
    // 保存下一条交易记录的变量
    Sales_item total;
    // 读入第一条交易记录，并确保有数据可以处理
    if (std::cin >> total)
    {
        // 保存和的变量
        Sales_item trans;
        // 读入并处理剩余交易记录
        while (std::cin >> trans)
        {
            // 连续处理相同的 ISBN 书籍
            if (total.isbn() == trans.isbn())
            {
                // 更新该书籍的总的销售额
                total += trans;
            }
            // 否则打印前一本书籍的结果
            else
            {
                std::cout << total << std::endl;
                // 更新迭代变量
                total = trans;
            }          
        }
        // while 结束后，打印最后一本书的结果
        std::cout << total << std::endl;        
    }
    // 用户没有任何输入，输出警告
    else
    {
        std::cerr << " No data?! " << std::endl;
        return -1;
    }    
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-25.cpp
** 3、运行生成的可执行程序，exercise 0< sale_item.txt; Ubuntu使用 ./exercise 0< sale_item.txt
** 4、技巧，使用 channel 0，直接将销售记录从文件中读取
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/