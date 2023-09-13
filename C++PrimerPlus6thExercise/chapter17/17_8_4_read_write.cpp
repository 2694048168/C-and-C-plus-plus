/**
 * @file 17_8_4_read_write.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

/**
 * @brief 编写C++程序, 将两个文件的对应每一行内容进行读取拼接后输出到一个新文件
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char *sourcefile1 = "./sourcefile1.txt";
    std::string sourcefile2 = "./sourcefile2.txt";
    std::string dst_file    = "./dst_file.txt";

    std::ifstream file_reader1(sourcefile1);
    std::ifstream file_reader2(sourcefile2);
    std::ofstream file_writer(dst_file);
    if (!file_writer.is_open() || !file_reader1.is_open() || !file_reader2.is_open())
    {
        std::cout << "open file is not successfully, please check.\n";
        return -1;
    }

    std::string line;
    while (!file_reader1.eof() || !file_reader2.eof())
    {
        std::stringstream buffer[2];
        
        if (std::getline(file_reader1, line))
        {
            buffer[0] << line;
        }

        if (std::getline(file_reader2, line))
        {
            buffer[1] << line << "\n";
        }

        file_writer << buffer[0].str() + buffer[1].str();
    }

    std::cout << "File operator successfully!\n";

    file_reader1.close();
    file_reader2.close();
    file_writer.close();

    return 0;
}