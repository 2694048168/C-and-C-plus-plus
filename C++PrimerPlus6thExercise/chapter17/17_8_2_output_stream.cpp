/**
 * @file 17_8_2_output_stream.cpp
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
 * @brief 编写C++程序, 将键盘的输入复制到指定的文件中(命令行指定)
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // 假设模拟的文件尾标志为 '$' '\n'
    std::string buffer;
    std::cout << "Please enter the character: \n";
    std::getline(std::cin, buffer);

    std::ofstream file_writer;
    file_writer.open(argv[1], std::ios::out);
    if (!file_writer.is_open())
    {
        std::cout << "open file is not successfully, please check.\n";
        std::cout << "====Use such as: " << argv[0] << " filepath\n";
        return -1;
    }

    for (const auto &elem : buffer)
    {
        if (elem == '$')
        {
            break;
        }

        file_writer << elem;
    }
    std::cout << "Writing into " << argv[1] << " successfully!\n";
    file_writer.close();

    return 0;
}