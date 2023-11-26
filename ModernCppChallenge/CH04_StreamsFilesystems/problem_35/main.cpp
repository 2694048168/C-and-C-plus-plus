/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

/**
 * @brief Computing the size of a directory
 * Write a function that computes the size of a directory, in bytes, recursively. 
 * It should be possible to indicate whether symbolic links should be followed or not.
 * ---------------------------------------------------------*/

/**
 * @brief Solution:

To compute the size of a directory, 
we have to iterate through all the files and sum the size of individual files.

std::filesystem::recursive_directory_iterator is an iterator from the filesystem
library that allows iterating all the entries of a directory in a recursive manner.
 It has various constructors, some of them taking a value of the type
std::filesystem::directory_options that indicates whether symbolic links 
 should be followed or not. 
 
The general purpose std::accumulate() algorithm can be used to 
sum together the file sizes. Since the total size of a directory could exceed 2 GB,
 you should not use int or long, but unsigned long long for the sum type.
------------------------------------------------------ */
// typedef unsigned long long uintmax_t;
std::uintmax_t get_directory_size(const std::filesystem::path folder, const bool follow_symlinks = false)
{
    auto iterator = std::filesystem::recursive_directory_iterator(
        folder, follow_symlinks ? std::filesystem::directory_options::follow_directory_symlink
                                : std::filesystem::directory_options::none);

    return std::accumulate(
        std::filesystem::begin(iterator), std::filesystem::end(iterator), 0ull,
        [](const std::uintmax_t total, const std::filesystem::directory_entry &entry)
        { return total + (std::filesystem::is_regular_file(entry) ? std::filesystem::file_size(entry.path()) : 0); });
}

// ------------------------------
int main(int argc, char **argv)
{
    std::cout << __cplusplus << '\n';

    if (argc != 2)
        throw std::runtime_error("Please enter the folder path");

    std::cout << "Size: " << get_directory_size(argv[1]) << " bytes\n";
    std::cout << "Size: " << get_directory_size(argv[1]) / (8 * 1024) << " KB\n";

    return 0;
}