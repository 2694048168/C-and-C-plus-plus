/* exercise 1-23、1-24
** 练习1.23 : 拷贝 Sales_item.hpp 到工作目录，
** 编写程序，读取多个销售记录，统计每个 isbn 有几条销售记录
** 练习1.24 : 输入不同的 isbn 测试程序
*/

#include <iostream>
#include "Sales_item.hpp"

int main()
{
    // 当前的 isbn，其 每一个 isbn 销售记录
    Sales_item current_Item, value_Item;
    // 读取 isbn 销售记录
    // EOF 结束符作为标识
    if (std::cin >> current_Item) 
    {
        // 统计销售记录条数
        int count_number = 1;
        while (std::cin >> value_Item) 
        {
            // 相同则统计记录条数加 1
            if (value_Item.isbn() == current_Item.isbn())
            {
                ++count_number;
            }
            // 否则输出结果，并更新下一次迭代循环统计
            else {
                std::cout << current_Item << " occurs " << count_number << " times " << std::endl;
                current_Item = value_Item;
                count_number = 1;
            }
        }
        // while 循环结束后
        // 输出统计结果
        std::cout << current_Item << " occurs " << count_number << " times " << std::endl;
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-23-24.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入例子 ：0-201-78345-x 3 20.00
**              0-201-78345-x 2 20.00
**              0-201-78345-x 2 20.00
**              0-202-78345-x 4 40.00
**              0-202-78345-x 1 40.00
**              ctrl + z, enter
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/