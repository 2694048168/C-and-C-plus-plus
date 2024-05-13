/**
 * @file 01_streamState.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief 流状态 Stream State
 * *流状态指示 I/O 操作是否失败, 每种流类型都以位集合的形式公开了用以表示状态的静态常量成员,
 *  它们指示可能的流状态: goodbit、badbit、eofbit 和 failbit.
 * 要确定流是否处于特定状态, 请调用返回布尔值的成员函数, 布尔值指示流是否处于相应状态.
 * TODO要重置流的状态, 可以调用其 clear() 方法.
 * 
 * 方法    | 状态    | 含义
 * good() | goodbit | 流处于好的工作状态
 * eof()  | eofbit  | 流遇到 EOF
 * fail() | failbit | 输入或输出操作失败, 但流仍可能处于好的工作状态
 * bad()  | badbit  | 遇到灾难性错误, 流状态不佳
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    //流实现了隐式的布尔转换(operator bool),可以简单直接地检查流是否处于良好的工作状态
    std::string word;
    size_t      count{0};

    /**
    * @brief 如何发送 EOF 取决于操作系统,
    * 1. 在 Windows 命令行中, 可以通过按下 ＜Ctrl+Z＞ 并按 ＜Enter＞ 来输入 EOF;
    * 2. 在 Linux bash 或 OS X shell 中, 按下 ＜Ctrl+D＞
    */
    while (std::cin >> word)
    {
        ++count;
    }
    std::cout << "Discovered " << count << " words.\n";

    /**
     * @brief 希望流在发出失败位时抛出异常,可以使用流的 exceptions 方法轻松完成此操作,
     * 该方法接受与要引发异常的位相对应的单个参数;
     * 如果需要多个位, 则可以简单地使用布尔"或"（|）将它们连接在一起;
     */
    std::cin.exceptions(std::istream::badbit);

    try
    {
        while (std::cin >> word) count++;
        std::cout << "Discovered " << count << " words.\n";
    }
    catch (const std::exception &exp)
    {
        std::cerr << "Error occurred reading from stdin: " << exp.what();
    }

    /**
     * @brief 缓冲和刷新 Buffering and Flushing
     * 许多 ostream 类模板在后台涉及操作系统调用, 例如,写入控制台,文件或网络socket;
     * 相对于其他函数调用, 系统调用通常很慢, 应用程序可以等待多个元素, 然后将它们一起发送以提高性能,
     * 而不是针对每个输出元素都调用系统调用.
     *
     * 排队行为称为缓冲(buffering), 当流清空缓冲的输出时, 称为刷新(flushing);
     * 通常, 此行为对用户是完全透明的, 但有时希望手动刷新 ostream, 可以求助于操纵符.
     */

    return 0;
}
