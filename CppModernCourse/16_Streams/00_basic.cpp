/**
 * @file 00_basic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <bitset>
#include <iostream>
#include <string>

/**
 * @brief Stream 流的基础知识
 * 使用这套标准的 stream 类框架可以获取各种各样的输入以及输出
 * 
 * 流可以对数据流进行建模, 在流中, 数据在对象之间流动, 这些对象可以对数据进行任意处理;
 * 当使用流时, 输入是进入流的数据; 输出是从流中出来的数据; 这些术语反映了用户对流的看待方法.
 * C++ 中, 流是执行输入和输出(I/O)的主要机制, 无论源或目标如何, 都可以使用流作为通用语言将输入连接到输出.
 * *STL 使用类继承机制来编码各种流类型之间的关系.
 * 
 * ?1. ＜ostream＞ 头文件中的 std::basic_ostream 类模板表示输出设备
 * ?2. ＜istream＞ 头文件中的 std::basic_istream 类模板表示输入设备
 * ?3. ＜iostream＞ 头文件中的 std::basic_iostream 类模板用于输入和输出设备
 * 三种流类型都需要两个模板参数，第一个对应于流的底层数据类型，第二个对应于特征类型.
 * 声明每种类型的头文件还为这些模板提供了 char 和 wchar_t 特化.(istream & wistream)
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 1. 全局流对象 Global Stream Objects
     * STL 在＜iostream＞头文件中提供了几个全局流对象, 它们包装了输入,输出和错误流
     * (stdin、stdout 和 stderr), 这些实现定义的标准流是程序与其执行环境之间的预连接通道,
     * 例如, 在桌面环境中, stdin 通常绑定到键盘, stdout 和 stderr 绑定到控制台.
     *
     * ====流类支持的操作可以分为两类:
     * ?1.格式化操作: 可能在执行 I/O 操作之前对其输入参数执行一些预处理;
     * ?2.未格式化操作: 直接执行 I/O 操作;
     *
     * TODO: 格式化操作
     * 所有格式化的 I/O 都通过两个函数: 标准流 operator＜< 和 operator＞＞,
     * !流以完全不相关的功能重载了左移右移运算符
     * 输出流重载 operator＜＜,  称为输出运算符或插入器, basic_ostream 类模板为所有基本类型
     *（void 和 nullptr_t 除 外）和一些STL容器(例 如 basic_string、complex 和 bitset)重载了输出运算符
     * *作为 ostream 用户, 不必担心这些重载如何将对象转换为可读输出
     */
    std::bitset<8> s{"01110011"};
    std::string    str("Crying zeros and I'm hearing ");
    size_t         num{111};

    std::cout << s << "\n" << str << num << "s\n";
    /**
     * @brief 标准流运算符的一个非常好的特性是它们通常返回对流的引用,
     * 从概念上讲, 重载通常按照以下方式定义:
     * ostream& operator＜＜(ostream&, char);
     * 这意味着可以将输出运算符连到一起, 使用这种技术
     *
     * 输入流重载 operator＞＞, 称为输入运算符或提取器,
     * basic_istream 类对所有与basic_ostream 相同类型的输入运算符都有相应的重载,
     * 同样, 作为用户可以在很大程度上忽略反序列化细节
     */
    double x, y;
    std::cout << "X: ";
    std::cin >> x;
    std::cout << "Y: ";
    std::cin >> y;
    std::string op;
    std::cout << "Operation: ";
    std::cin >> op;

    if (op == "+")
    {
        std::cout << x + y;
    }
    else if (op == "-")
    {
        std::cout << x - y;
    }
    else if (op == "*")
    {
        std::cout << x * y;
    }
    else if (op == "/")
    {
        std::cout << x / y;
    }
    else
    {
        std::cout << "Unknown operation " << op << std::endl;
    }

    /**
     * @brief 未格式化操作 Unformatted Operations
     * 当使用基于文本的流时, 通常需要使用格式化运算符;
     * *但是, 如果正在处理二进制数据, 或者正在编写需要对流进行底层访问的代码, 那么需要了解未格式化操作,
     * 未格式化的 I/O 操作涉及很多细节,
     * *istream 类有许多未格式化的输入方法, 这些方法在字节级别操作流;
     * *输出流必然支持未格式化写操作, 这些操作在非常低的级别上操作流;
     *
     * TODO: https://en.cppreference.com/w/cpp/io
     *
     * ====基本类型的特殊格式 Special Formatting for Fundamental Types
     * 除了 void 和 nullptr 之外, 所有基本类型都有输入和输出运算符重载, 但有些有特殊规则:
     * 1. char 和 wchar_t: 输入运算符在分配字符类型时会跳过空格;
     * 2. char* 和 wchar_t*: 输入运算符首先跳过空格, 然后读取字符串, 
     *         直到遇到另一个空格或文件结尾(End-Of-File,EOF), 必须为输入保留足够的空间.
     * 3. void*: 地址格式取决于输入和输出运算符的实现, 在桌面系统上,地址采用十六进制字面量形式,
     *            例如 32 位的 0x01234567 或 64 位的 0x0123456789abcdef.
     * 4. bool: 输入和输出运算符将布尔值视为数字: 1 表示 true, 0 表示 false;
     * 5. 数值类型: 输入运算符要求输入至少以一位数字开头, 格式错误的输入数字会产生零值结果.
     *
     */

    return 0;
}
