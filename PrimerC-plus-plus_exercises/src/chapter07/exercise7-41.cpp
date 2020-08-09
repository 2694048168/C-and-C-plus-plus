/* exercise 7-41
** 练习7.41: 使用委托构造函数重新编写你的Sales_data 类，
** 给每个构造函数体添加一条语句，令其一旦执行就打印一条信息。
** 用各种可能的方式分别创建Sales_data对象，认真研究每次输出的信息直到你确实理解了委托构造函数的执行顺序。
**
*/

#include "exercise7-41.hpp"

// constructor
Sales_data::Sales_data(std::istream &is) : Sales_data()
{
    std::cout << "Sales_data(istream &is)" << std::endl;
    read(is, *this);
}

// member functions.
Sales_data& Sales_data::combine(const Sales_data& rhs)
{
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

// friend functions
std::istream &read(std::istream &is, Sales_data &item)
{
    double price = 0;
    is >> item.bookNo >> item.units_sold >> price;
    item.revenue = price * item.units_sold;
    return is;
}

std::ostream &print(std::ostream &os, const Sales_data &item)
{
    os << item.isbn() << " " << item.units_sold << " " << item.revenue;
    return os;
}

Sales_data add(const Sales_data &lhs, const Sales_data &rhs)
{
    Sales_data sum = lhs;
    sum.combine(rhs);
    return sum;
}


int main(int argc, char **argv)
{
    std::cout << "1. default way: " << std::endl;
    std::cout << "----------------" << std::endl;
    Sales_data s1;   
    
    std::cout << "\n2. use std::string as parameter: " << std::endl;
    std::cout << "----------------" << std::endl;
    Sales_data s2("CPP-Primer-5th");
    
    std::cout << "\n3. complete parameters: " << std::endl;
    std::cout << "----------------" << std::endl;
    Sales_data s3("CPP-Primer-5th", 3, 25.8);
    
    std::cout << "\n4. use istream as parameter: " << std::endl;
    std::cout << "----------------" << std::endl;
    Sales_data s4(std::cin);
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-41.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
