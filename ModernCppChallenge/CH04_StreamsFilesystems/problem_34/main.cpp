/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Removing empty lines from a text file
 * Write a program that, given the path to a text file, 
 * modifies the file by removing all empty lines. 
 * Lines containing only whitespaces are considered empty.
 * ---------------------------------------------------------*/

/**
 * @brief Solution:
 A possible approach to solving this task is to do the following:
1. Create a temporary file to contain only the text 
    you want to retain from the original file;
2. Read line by line from the input file and copy to the temporary file all lines 
    that are not empty;
3. Delete the original file after finishing processing it;
4. Move the temporary file to the path of the original file;

An alternative is to move the temporary file and overwrite the original one. 
The following implementation follows the steps listed. 
The temporary file is created in the temporary directory returned by 
    std::filesystem::temp_directory_path():
------------------------------------------------------ */
void remove_empty_lines(std::filesystem::path filepath)
{
    std::ifstream file_in(filepath.native(), std::ios::in);
    if (!file_in.is_open())
        throw std::runtime_error("cannot open input file");

    auto temp_path = std::filesystem::temp_directory_path() / "temp.txt";

    std::ofstream file_out(temp_path.native(), std::ios::out | std::ios::trunc);
    if (!file_out.is_open())
        throw std::runtime_error("cannot create temporary file");

    std::string line;
    while (std::getline(file_in, line))
    {
        // 判断该行不是空行
        if (line.length() > 0 && line.find_first_not_of(' ') != line.npos)
        {
            file_out << line << '\n';
        }
    }

    file_in.close();
    file_out.close();

    std::filesystem::remove(filepath);
    std::filesystem::rename(temp_path, filepath);
}

// ------------------------------
int main(int argc, char **argv)
{
    std::cout << __cplusplus << '\n';

    remove_empty_lines("sample34.txt");
    std::cout << "The empty lines of this file is remove\n";

    return 0;
}
