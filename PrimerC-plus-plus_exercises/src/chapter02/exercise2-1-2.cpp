/* exercise 2-1、2-2
** 练习2.1: 类型 int、long、 long long和short 区别是什么？
** 无符号类型和带符号类型区别是什么？
** float 和 double 区别是什么？
** solution:
** step0 : 定点数整数，在计算机中储存所占字节大小不一样
** step1 ：无符号和有符号整数，其表示的整数范围不一样
** step2 ： float 和 double 浮点数在计算机中储存时保留的精度不一样
** summary：根据系统的架构不一样，系统字长等等区别，有一定差异
**
** 练习2.2: 计算按揭贷款时，对于利率、本金和付款分别选择何种数据类型，说明理由
** solution：对于金融数据而言，对于数据的精度要求极高，最好用高精度 double 类型
*/

#include <iostream>

int main()
{
    // 带符号类型
    short short_value = 0;
    int int_value = 0;
    long long_value = 0;
    long long long_long_value = 0;
    float float_value = 0.0;
    double double_value = 0.00;

    // 无符号类型
    unsigned short unsigned_short_value = 0;
    unsigned int unsigned_int_value = 0;
    unsigned long unsigned_long_value = 0;
    unsigned long long unsigned_long_long_value = 0;
    
    // 浮点数，储存方式不同，不同这样组合显示无符号情况
    // unsigned float float_value = 0.0;
    // unsigned double double_value = 0.00;

    std::cout << sizeof(short_value) << " Byte "
              << sizeof(int_value) << " Byte "
              << sizeof(long_value) << " Byte "
              << sizeof(long_long_value) << "Byte  "
              << sizeof(float_value) << " Byte "
              << sizeof(double_value) << " Byte "
              << sizeof(unsigned_short_value) << " Byte "
              << sizeof(unsigned_int_value) << " Byte "
              << sizeof(unsigned_long_value) << " Byte "
              << sizeof(unsigned_long_long_value) << " Byte "
              << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-1-2.cpp
** 3、编译源代码文件并指定标准版本，g++ --version; g++ -std=c++11 -o exercise exercise2-1-2.cpp
** 4、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
