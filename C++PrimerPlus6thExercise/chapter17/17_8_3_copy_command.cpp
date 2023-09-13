/**
 * @file 17_8_3_copy_command.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief 编写C++程序, 实现 Linux copy command 效果
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please check usage\n";
        std::cout << "====Use such as: " << argv[0] << " source-filepath dest-filepath\n";

        return -1;
    }

    std::ifstream file_reader(argv[1], std::ios::binary);
    std::ofstream file_writer(argv[2], std::ios::binary);
    if (!file_writer.is_open() || !file_reader.is_open())
    {
        std::cout << "open file is not successfully, please check.\n";
        return -1;
    }

    file_writer << file_reader.rdbuf();

    // -----------------------------------------------------
    // std::string buffer;
    // while (!file_reader.eof())
    // // char        ch;
    // // while ((ch = file_reader.get()) != EOF)
    // {
    //     // std::getline(file_reader, buffer);
    //     file_reader >> buffer;
    //     file_writer << buffer;
    // }
    // ----------------------------------------------------
    
    std::cout << "\nCopy " << argv[1] << "file into " << argv[2] << " successfully!\n";

    file_reader.close();
    file_writer.close();

    return 0;
}