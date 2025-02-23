/**
 * @file path.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 路径对象 path 是文件系统的基石, 其设计体现了类型安全的精髓
 * @version 0.1
 * @date 2025-02-23
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ path.cpp -std=c++20
 * clang++ path.cpp -std=c++20
 * 
 */

#include <filesystem>
#include <iostream>

// --------------------------------------
int main(int argc, const char *argv[])
{
    // 路径操作的艺术
    std::filesystem::path p1 = "C:\\Program Files";  // Windows风格
    std::filesystem::path p2 = "/var/log/app.log";   // UNIX风格
    std::filesystem::path p3 = R"(D:\Data\2023\07)"; // Raw string处理特殊字符

    // 智能路径拼接
    std::filesystem::path full_path = p3 / "report.csv"; // 自动处理路径分隔符
    std::cout << full_path << std::endl;
    full_path = p3.append("report.csv"); // 自动处理路径分隔符
    std::cout << full_path << std::endl;
    full_path = p3.concat("report.csv"); // 自动处理路径分隔符
    std::cout << full_path << std::endl;

    // 路径规范化
    std::filesystem::path messy_path = "D:/Data//./tmp/../config.xml";
    std::cout << messy_path.lexically_normal() << std::endl;

    std::cout << "=========================================\n";
    // 文件状态探秘
    // file_status 和 directory_entry 构成了元数据操作的双子星
    auto check_file = [](const std::filesystem::path &p)
    {
        if (!std::filesystem::exists(p))
        {
            std::cerr << p.string() << " Path not found\n" << std::endl;
            return;
        }
        std::filesystem::file_status s = std::filesystem::status(p);
        std::cout << "Type: ";
        switch (s.type())
        {
        case std::filesystem::file_type::regular:
            std::cout << "Regular File";
            break;
        case std::filesystem::file_type::directory:
            std::cout << "Directory";
            break;
        case std::filesystem::file_type::symlink:
            std::cout << "Symbolic Link";
            break;
        default:
            std::cout << "Other";
        }
        std::cout << "\nSize: " << std::filesystem::file_size(p) << " bytes" << std::endl;
    };

    std::filesystem::path test_file = "data.bin";
    check_file(test_file);
    test_file = "path.cpp";
    check_file(test_file);

    std::cout << "=========================================\n";
    // 目录遍历的现代范式
    // directory_iterator 和 recursive_directory_iterator 提供了强大的遍历能力
    auto analyze_directory = [](const std::filesystem::path &dir) -> void
    {
        if (!std::filesystem::is_directory(dir))
            return;

        uintmax_t total_size = 0;
        size_t    file_count = 0;
        for (const auto &entry : std::filesystem::recursive_directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                total_size += entry.file_size();
                ++file_count;
            }
        }
        std::cout << "Directory analysis:\n"
                  << "Total files: " << file_count << "\n"
                  << "Total size: " << total_size << " bytes (" << (total_size / (1024.0 * 1024)) << " MB)"
                  << std::endl;
    };

    analyze_directory(std::filesystem::current_path());

    return 0;
}
