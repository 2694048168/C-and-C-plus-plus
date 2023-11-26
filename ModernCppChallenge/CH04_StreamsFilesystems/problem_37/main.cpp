/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string_view>
#include <vector>

/**
 * @brief Finding files in a directory that match a regular expression
 * 
 * Write a function that, given the path to a directory and a regular expression, 
 * returns a list of all the directory entries whose names match the regular expression.
 * ---------------------------------------------------------*/

/**
 * @brief Solution:

Implementing the specified functionality should be straightforward: 
 iterate recursively through all the entries of the specified directory 
 and retain all the entries that are regular files 
 and whose name matches the regular expression. 

To do that, you should use the following:
1. std::filesystem::recursive_directory_iterator to iterate through directory entries
2. std::regex and std::regex_match() to check whether the filename matches the regular expression
3. copy_if() and back_inserter to copy, at the end of a vector,
  the directory entries that match a specific criteria.
------------------------------------------------------ */
std::vector<std::filesystem::directory_entry> findFiles(const std::filesystem::path &path,
                                                        const std::string_view      &regex)
{
    std::vector<std::filesystem::directory_entry> result;

    std::regex rx(regex.data());

    std::copy_if(std::filesystem::recursive_directory_iterator(path), std::filesystem::recursive_directory_iterator(),
                 std::back_inserter(result),
                 [&rx](const std::filesystem::directory_entry &entry) {
                     return std::filesystem::is_regular_file(entry.path())
                         && std::regex_match(entry.path().filename().string(), rx);
                 });

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    // auto dir = std::filesystem::temp_directory_path();
    // auto pattern = R"(wct[0-9a-zA-Z]{3}\.tmp)";
    // auto result  = findFiles(dir, pattern);

    auto folder  = std::filesystem::path("./");
    auto pattern = R"(wct[0-9a-zA-Z]{3}.cpp)";
    auto result  = findFiles(folder, pattern);

    for (const auto &entry : result)
    {
        std::cout << entry.path().string() << std::endl;
    }

    return 0;
}