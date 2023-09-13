/**
 * @file 17_8_5_file_operating.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief 编写C++程序, 利用文件读写操作完成程序任务
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char *sourcefile1 = "./mat.dat";
    const char *sourcefile2 = "./pat.dat";
    const char *dst_file    = "./matpat.dat";

    std::ifstream file_reader1(sourcefile1);
    std::ifstream file_reader2(sourcefile2);
    std::ofstream file_writer(dst_file);
    if (!file_writer.is_open() || !file_reader1.is_open() || !file_reader2.is_open())
    {
        std::cout << "open file is not successfully, please check.\n";
        return -1;
    }

    std::vector<std::string> mat_vec;
    std::vector<std::string> pat_vec;

    std::string line;
    while (!file_reader1.eof() || !file_reader2.eof())
    {
        // std::stringstream buffer[2];

        if (std::getline(file_reader1, line))
        {
            // buffer[0] << line;
            // mat_vec.push_back(buffer[0].str());
            mat_vec.push_back(line);
        }

        if (std::getline(file_reader2, line))
        {
            pat_vec.push_back(line);
        }
    }

    auto print = [](std::vector<std::string> &vec)
    {
        std::cout << "------------------------------\n";
        for (const auto &elem : vec)
        {
            std::cout << "Name: " << elem << "\t";
        }
        std::cout << "\n------------------------------\n";
    };

    print(mat_vec);
    std::cout << "Mat has " << mat_vec.size() << " friends to here\n";

    print(pat_vec);
    std::cout << "Pat has " << pat_vec.size() << " friends to here\n";

    mat_vec.insert(mat_vec.end(), pat_vec.begin(), pat_vec.end());
    std::sort(mat_vec.begin(), mat_vec.end());
    auto last_iter = std::unique(mat_vec.begin(), mat_vec.end());
    mat_vec.erase(last_iter, mat_vec.end());
    for (const auto &elem : mat_vec)
    {
        file_writer << elem << "\n";
    }
    std::cout << "Mat and Pat have " << mat_vec.size() << " friends to here\n";

    std::cout << "File operator successfully!\n";

    file_reader1.close();
    file_reader2.close();
    file_writer.close();

    return 0;
}