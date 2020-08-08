/* exercise 1-21
** 练习1.21 : 拷贝 Sales_item.hpp 到工作目录，
** 编写程序，读取两个 ISBN 相同书籍的销售记录，输出其和
*/

#include <iostream>
#include "Sales_item.hpp"

int main()
{
    Sales_item item1, item2;
    // solution 1
    // EOF
    std::cin >> item1 >> item2;
    // 如果具有相同的 ISBN 则相加
    if (item1.isbn() == item2.isbn())
    {
        std::cout << item1 + item2 << std::endl;
        return 0;
    }
    // 否则提示错误
    else 
    {
        std::cerr << "Data must refer to same ISBN." << std::endl;
        return -1;
    }
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-21.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入例子 ：0-201-78345-x 3 20.00
**              0-201-78345-x 2 20.00
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
