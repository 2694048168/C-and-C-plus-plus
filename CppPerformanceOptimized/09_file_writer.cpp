/**
 * @file 09_file_writer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <fstream>
#include <iostream>
#include <string>

void stream_write_line(std::ostream &f, const std::string &line)
{
    f << line << std::endl;
}

// 如果没有 std::endl, 那么写文件应当会快很多,
// 因为 std::ofstream 只是将几个大数据块传递给了操作系统
void stream_write_line_noflush(std::ostream &f, const std::string &line)
{
    f << line << '\n';
}

// --------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "The optimized for I/O\n";

    return 0;
}
