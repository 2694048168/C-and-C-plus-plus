/**
 * @file 04_recursiveDirectoryIteration.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <filesystem>
#include <iostream>
#include <string_view>

/**
 * @brief Recursive Directory Iteration 递归目录迭代
 * 
 * 使用 std::filesystem::recursive_directory_iterator 列出
 * 给定路径的子目录的文件数和总大小的程序
 */

struct Attributes
{
    Attributes &operator+=(const Attributes &other)
    {
        this->size_bytes += other.size_bytes;
        this->n_directories += other.n_directories;
        this->n_files += other.n_files;
        return *this;
    }

    size_t size_bytes;
    size_t n_directories;
    size_t n_files;
};

void print_line(const Attributes &attributes, std::string_view path)
{
    std::cout << std::setw(14) << attributes.size_bytes << std::setw(7) << attributes.n_files << std::setw(7)
              << attributes.n_directories << " " << path << "\n";
}

Attributes explore(const std::filesystem::directory_entry &directory)
{
    Attributes attributes{};

    for (const auto &entry : std::filesystem::recursive_directory_iterator{directory.path()})
    {
        if (entry.is_directory())
        {
            attributes.n_directories++;
        }
        else
        {
            attributes.n_files++;
            attributes.size_bytes += entry.file_size();
        }
    }

    return attributes;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: tree-dir PATH";
        return -1;
    }

    const std::filesystem::path sys_path{argv[1]};

    std::cout << "Size Files Dirs Name\n";
    std::cout << "-------------- ------ ------ ------------\n";
    Attributes root_attributes{};

    for (const auto &entry : std::filesystem::directory_iterator{sys_path})
    {
        try
        {
            if (entry.is_directory())
            {
                const auto attributes = explore(entry);
                root_attributes += attributes;
                print_line(attributes, entry.path().string());
                root_attributes.n_directories++;
            }
            else
            {
                root_attributes.n_files++;
                std::error_code ec;
                root_attributes.size_bytes += entry.file_size(ec);
                if (ec)
                    std::cerr << "Error reading file size: " << entry.path().string() << std::endl;
            }
        }
        catch (const std::exception &)
        {
        }
    }

    print_line(root_attributes, argv[1]);

    return 0;
}
