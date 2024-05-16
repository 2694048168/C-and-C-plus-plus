/**
 * @file 03_directoryIterators.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <filesystem>
#include <iomanip>
#include <iostream>

/**
 * @brief 目录迭代器 Directory Iterators
 * 文件系统库提供了两个用于迭代目录元素的类: 
 * *std::filesystem::directory_iterator
 * *std::filesystem::recursive_directory_iterator
 * directory_iterator 不会进入子目录, 但 recursive_directory_iterator 会.
 * 
 * ====directory_iterator 的默认构造函数产生结束迭代器;
 * 另一个构造函数接受路径参数, 它指示要枚举的目录;
 * 此外, 也可以提供 std::filesystem::directory_options, 它是具有以下常量的枚举类:
 * 1. none 指示迭代器跳过目录符号链接, 如果迭代器遇到权限拒绝,则会产生错误;
 * 2. follow_directory_symlink 遵循符号链接;
 * 3. 如果迭代器遇到权限拒绝的情况, skip_permission_denied 会跳过目录;
 *   此外可以提供一个 std::error_code, 它与所有其他接受 error_code 的文件系统库函数一样,
 *  将设置此参数, 而不是在构造过程中发生错误时抛出异常.
 * 
 * ====目录条目 Directory Entries
 * 输入迭代器 directory_iterator 和 recursive_directory_iterator 为每个条目生成一个
 * std::filesystem::directory_entry 元素; 
 * directory_entry 类存储路径, 以及这些路径的一些属性, 并把它们暴露为可使用的方法.
 * 
 * 
 */
void describe(const std::filesystem::directory_entry &entry)
{
    try
    {
        if (entry.is_directory())
        {
            std::cout << " *";
        }
        else
        {
            std::cout << std::setw(12) << entry.file_size();
        }
        const auto lw_time = duration_cast<std::chrono::seconds>(entry.last_write_time().time_since_epoch());
        std::cout << std::setw(12) << lw_time.count() << " " << entry.path().filename().string() << "\n";
    }
    catch (const std::exception &exp)
    {
        std::cout << "Error accessing " << entry.path().string() << ": " << exp.what() << std::endl;
    }
}

// -----------------------------------
int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: list-dir PATH";
        return -1;
    }

    const std::filesystem::path sys_path{argv[1]};
    std::cout << "Size Last Write Name\n";
    std::cout << "------------ ----------- ------------\n";

    for (const auto &entry : std::filesystem::directory_iterator{sys_path})
    {
        describe(entry);
    }

    return 0;
}
