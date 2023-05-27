/**
 * @file file_handling.cpp
 * @author Wei Li (weili@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief the file handing in C++ with fstream class.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* ------------------------------------ */
    std::ofstream fout;
    const char   *filepath = "sample.txt";
    fout.open(filepath);
    if (fout.fail())
    {
        std::cout << "open the " << filepath << " failed\n";
    }

    std::string line;
    while (fout)
    {
        std::getline(std::cin, line);
        if (line == "-1")
        {
            break;
        }
        fout << line << "\n" << std::endl;
    }
    fout.close();

    /* ------------------------------------ */
    std::ifstream fin;
    fin.open(filepath);
    if (fout.fail())
    {
        std::cout << "open the " << filepath << " failed\n";
    }

    while (std::getline(fin, line))
    {
        std::cout << line << std::endl;
    }
    fin.close();

    /* ------------------------------------ */
    // TODO rename file
    // TODO remove file
    // TODO get file size
    // TODO check file exists

    return 0;
}
