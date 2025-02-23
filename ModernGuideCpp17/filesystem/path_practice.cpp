/**
 * @file path_practice.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-23
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ path_practice.cpp -std=c++20
 * clang++ path_practice.cpp -std=c++20
 * 
 */

#include <filesystem>
#include <iostream>

/**
 * @brief 工业级应用实战-智能文件同步器
 * 
 * @param src 
 * @param dst 
 */
void sync_files(const std::filesystem::path &src, const std::filesystem::path &dst)
{
    auto copy_options = std::filesystem::copy_options::update_existing | std::filesystem::copy_options::recursive;

    try
    {
        // 创建目标目录结构
        std::filesystem::create_directories(dst);
        for (const auto &entry : std::filesystem::recursive_directory_iterator(src))
        {
            const auto relative_path = std::filesystem::relative(entry.path(), src);
            const auto target_path   = dst / relative_path;
            if (entry.is_directory())
            {
                std::filesystem::create_directory(target_path);
            }
            else if (entry.is_regular_file())
            {
                // 仅复制更新的文件
                if (!std::filesystem::exists(target_path)
                    || entry.last_write_time() > std::filesystem::last_write_time(target_path))
                {
                    std::filesystem::copy_file(entry.path(), target_path, copy_options);
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error &excep)
    {
        std::cerr << "Sync error: " << excep.what() << std::endl;
    }
}

/**
 * @brief 工业级应用实战-安全文件删除工具
 * 
 * @param target 
 */
void secure_remove(const std::filesystem::path &target)
{
    try
    {
        if (std::filesystem::is_directory(target))
        {
            // 递归计算目录大小
            uintmax_t size = 0;
            for (const auto &entry : std::filesystem::recursive_directory_iterator(target))
            {
                if (entry.is_regular_file())
                    size += entry.file_size();
            }
            std::cout << "Deleting directory (" << (size / (1024.0 * 1024)) << " MB)...";
            std::filesystem::remove_all(target);
            std::cout << " Done." << std::endl;
        }
        else if (std::filesystem::is_regular_file(target))
        {
            std::cout << "Deleting file (" << std::filesystem::file_size(target) << " bytes)...";
            std::filesystem::remove(target);
            std::cout << " Done." << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error &excep)
    {
        std::cerr << "Deletion failed: " << excep.what() << std::endl;
    }
}

// =====================================
int main(int argc, const char *argv[])
{
    const std::filesystem::path src_path = ".";
    const std::filesystem::path dst_path = "../back/";
    sync_files(src_path, dst_path);
    secure_remove(dst_path);

    return 0;
}
