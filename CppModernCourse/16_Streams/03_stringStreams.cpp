/**
 * @file 03_stringStreams.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <sstream>
#include <string>

/**
 * @brief 字符串流 string stream
 * ?字符串流(string stream)类为读取和写入字符序列提供了便利.
 * 这些类在几种情况下很有用, 如果要将字符串数据解析为类型, 则输入字符串特别有用,
 * 因为可以使用输入运算符, 所以可以使用所有标准操纵符工具;
 * 输出字符串非常适合从可变长度输入构建字符串;
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 输出字符串流
     * 输出字符串流为字符序列提供输出流语义, 派生自＜sstream＞头文件中的
     * 类模板 std::basic_ostringstream, 它提供以下特化:
     * *using ostringstream = basic_ostringstream＜char＞;
     * *using wostringstream = basic_ostringstream＜wchar_t＞;
     * 输出字符串流支持与 ostream 相同的所有功能, 每当将输入发送到字符串流时, 
     * 该流都会将此输入存储到内部缓冲区中, 
     * ?可以认为这在功能上等同于字符串的 append 操作(只不过字符串流可能更高效).
     *
     * 输出字符串流还支持 str() 方法, 该方法有两种操作模式.
     * 1. 如果没有参数, str 返回内部缓冲区的副本, 将其作为 basic_string
     * (因此 ostringstream 返回一个 string; wostringstream 返回一个 wstring).
     * 2. 给定一个 basic_string 参数, 字符串流将用参数的内容替换其缓冲区的当前内容.
     * 
     */
    std::cout << "[====]ostringstream produces strings with str\n";
    std::ostringstream ss;
    ss << "By Grab thar's hammer, ";
    ss << "by the suns of Worn. ";
    ss << "You shall be avenged.";

    const auto lazarus = ss.str();
    std::cout << lazarus << '\n';

    ss.str("I am Good.");
    const auto Good = ss.str();
    std::cout << Good << '\n';

    /**
     * @brief 2. 输入字符串流
     * 输入字符串流为字符序列提供输入流语义, 派生自＜sstream＞头文件中的
     * 类模板 std::basic_istringstream, 它提供以下特化:
     * *using istringstream = basic_istringstream＜char＞;
     * *using wistringstream = basic_istringstream＜wchar_t＞;
     * ?这些类似于 basic_ostringstream 特化,
     *  可以通过传递具有适当特化的 basic_string 来构造输入字符串流(string 用于 istringstream,
     *  wstring 用 于 wistringstream).
     * 
     */
    std::cout << "[====]istringstream supports construction from a string\n";
    std::string        numbers("1 2.23606 2");
    std::istringstream ss_in{numbers};

    int   a;
    float b, c, d{};
    ss_in >> a;
    ss_in >> b;
    ss_in >> c;

    std::cout << a << ' ' << b << ' ' << c << '\n';
    ss_in >> d;
    std::cout << d << '\n';

    /**
     * @brief 3. 支持输入和输出的字符串流
     * 如果想要支持输入和输出操作的字符串流, 则可以使用 basic_stringstream,
     * *using stringstream = basic_stringstream＜char＞;
     * *using wstringstream = basic_stringstream＜wchar_t＞;
     * 此类支持输入和输出运算符、str 方法以及从字符串构造
     * 
     */
    std::cout << "[====]stringstream supports all string stream operations\n";
    std::stringstream ss_in_out;
    ss << "Zed's DEAD";
    std::string who;
    ss_in_out >> who;

    int what;
    ss_in_out >> std::hex >> what;

    std::cout << who << '\n';
    std::cout << what << '\n';

    // std::basic_stringstream 的部分操作
    // TODO: https://en.cppreference.com/w/cpp/io/basic_stringstream

    return 0;
}
