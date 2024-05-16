/**
 * @file 05_fstreamInteroperation.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>

/**
 * @brief fstream 互操作
 * 除了字符串类型之外, 还可以使用 std::filesystem::path 或 std::filesystem::directory_entry
 *  构造文件流(basic_ifstream、basic_ofstream 或 basic_fstream).
 * 
 * 例如 可以遍历一个目录并构造一个 ifstream 来读取遇到的每个文件,
 * 如何检查每个 Windows 可移植的可执行文件(.sys、.dll、.exe 等)开的特别的 MZ 字节, 并报告违反此规则的文件.
 */

// -------------------------------------
int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: pe-check PATH\n";
        return -1;
    }

    const std::unordered_set<std::string> pe_extensions{".acm", ".ax",  ".cpl", ".dll", ".drv", ".efi",
                                                        ".exe", ".mui", ".ocx", ".scr", ".sys", ".tsp"};

    const std::filesystem::path sys_path{argv[1]};
    std::cout << "Searching " << sys_path << " recursively.\n";
    size_t n_searched{};

    auto iterator = std::filesystem::recursive_directory_iterator{
        sys_path, std::filesystem::directory_options::skip_permission_denied};

    for (const auto &entry : iterator)
    {
        try
        {
            if (!entry.is_regular_file())
                continue;

            const auto &extension = entry.path().extension().string();
            const auto  is_pe     = pe_extensions.find(extension) != pe_extensions.end();

            if (!is_pe)
                continue;

            std::ifstream file{entry.path()};
            char          first{}, second{};
            if (file)
                file >> first;
            if (file)
                file >> second;

            if (first != 'M' || second != 'Z')
                std::cout << "Invalid PE found: " << entry.path().string() << "\n";

            ++n_searched;
        }
        catch (const std::exception &exp)
        {
            std::cerr << "Error reading " << entry.path().string() << ": " << exp.what() << '\n';
        }
    }

    std::cout << "Searched " << n_searched << " PEs for magic bytes.\n";

    return 0;
}
