/**
 * @file 6_11_8_read_file.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <fstream>
#include <iostream>

/**
 * @brief 编写C++程序, 读取文件并统计文件的字符数量
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // TODO 注意文件路径和文件名的区别, 是否需要对路径和文件名进行拼接操作?
    const char *filename = "./test.txt";

    std::ifstream file_reader;
    file_reader.open(filename, std::ios::in);
    if (!file_reader.is_open())
    {
        std::cout << "read file is not successfully, please check." << filename << "\n";
        return -1;
    }

    // std::string buffer;
    char         buffer;
    /* '\0' 怎么处理, 会出现多读一个字符的情况;
    如何理解并准确处理 C++ 中 EOF 标志位
    ----------------------------------- */
    unsigned int num_chars = 0;
    // while (!file_reader.eof())
    while ((buffer = file_reader.get()) != EOF)
    {
        // file_reader >> buffer;
        // std::cout << buffer << "\n";
        if (buffer != ' ')
        {
            ++num_chars;
        }
    }

    file_reader.close();
    /* ------------------------------------- */
    std::cout << "The total number of character is: " << num_chars << std::endl;

    return 0;
}