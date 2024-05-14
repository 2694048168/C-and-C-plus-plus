/**
 * @file 05_streamBuffers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <fstream>
#include <iostream>
#include <iterator>

/**
 * @brief 流缓冲区 Stream Buffers
 * 流不直接读写, 它在后台使用流缓冲区类, 概括地说, 流缓冲区类是发送或提取字符的模板.
 * 除非实现自己的流库, 否则实现细节并不重要, 但重要的是要知道它们存在于多种上下文中.
 * ?获取流缓冲区的方法是使用流的 rdbuf 方法, 所有流都提供该方法.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 1. 将文件写入 stdout
     * 有时只想将输入文件流的内容直接写入输出流, 为此可以从文件流中提取流缓冲区指针并将其传递给输出运算符,
     * 例如可以通过以下方式使用 cout 将文件的内容转储到 stdout:
     * *std::cout ＜＜ my_ifstream.rdbuf();
     * 
     */
    std::ifstream my_ifstream{"lunchtime.txt"};
    std::cout << my_ifstream.rdbuf() << '\n';

    /**
     * @brief 2. 输出流缓冲区迭代器
     * 输出流缓冲区迭代器是模板类, 它公开了一个输出迭代器接口,
     * 该接口将写入操作转换为底层流缓冲区上的输出操作.
     * 换句话说, 这些适配器允许像使用输出迭代器一样使用输出流.
     * 
     * 要构造输出流缓冲区迭代器, 请使用＜iterator＞头文件中的 ostreambuf_iterator 模板类,
     * 它的构造函数接受一个输出流参数和一个对应于构造函数参数的模板参数(字符类型)
     */
    std::ostreambuf_iterator<char> itr{std::cout};
    *itr = 'H';
    ++itr;
    *itr = 'i';

    /**
     * @brief 3. 输入流缓冲区迭代器
     * 输入流缓冲区迭代器是模板类, 它公开了一个输入迭代器接口, 
     * 该接口将读取操作转换为对底层流缓冲区的读取操作, 这些完全类似于输出流缓冲区迭代器.
     *
     * 要构造输入流缓冲区迭代器, 请使用＜iterator＞头文件中的 istreambuf_iterator模板类,
     * 与 ostreambuf_iterator 不同, 它采用流缓冲区参数, 因此必须在要适应的输入流上调用 rdbuf(),
     *  此参数是可选的: istreambuf_iterator 的默认构造函数对应于输入迭代器的范围结束迭代器.
     * 
     */
    std::istreambuf_iterator<char> cin_itr{std::cin.rdbuf()}, end{};
    std::cout << "What is your name? ";
    const std::string name{cin_itr, end};
    std::cout << "\nGoodbye, " << name << '\n';

    return 0;
}
