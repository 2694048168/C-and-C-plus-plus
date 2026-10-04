/**
 * @file 27_filesystem.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-09-22
 * 
 * @copyright Copyright (c) 2026
 * 
 * https://changkun.de/modern-cpp/zh-cn/08-filesystem/
 * 
 */

#include <windows.h>

/**
 * @brief 文件系统库提供了文件系统、路径、常规文件、目录等等相关组件进行操作的相关功能.
 * Modern C++17 成为 C++ 标准,头文件<filesystem> 中的 std::filesystem 命名空间.
 * 
 * std::filesystem::path 是整个库的核心，它以一种可移植的方式表示文件路径，
 * 并屏蔽了不同操作系统在路径分隔符（如 / 与 \）上的差异。
 * path 仅仅是对路径的语法表示，构造一个 path 并不会访问磁盘，
 * 也不要求该路径真实存在。它提供了一组用于分解路径的成员函数。
 * 
 * 查询文件状态,非成员函数用于查询路径对应的实际文件,
 * 会真正访问文件系统的操作在出错时（例如路径不存在、权限不足）
 * 会抛出 std::filesystem::filesystem_error 异常。
 * 库为几乎每个此类函数都提供了一个接受 std::error_code& 的重载版本，
 * 用于以非异常的方式获取错误
 * 
 */
#include <filesystem>
// namespace fs = std::filesystem;
#include <fstream>
#include <iostream>
#include <print>
#include <string>

void func()
{
    // 在系统临时目录下创建一个专用的工作目录，使示例自包含且可重复运行
    const std::filesystem::path base = std::filesystem::temp_directory_path() / "modern-cpp-fs-demo";
    std::filesystem::remove_all(base);                 // 清理上一次运行的残留
    std::filesystem::create_directories(base / "sub"); // 递归创建中间目录

    // 创建一个文件
    std::ofstream(base / "hello.txt") << "hello, filesystem";

    // 路径分解（不访问磁盘）
    const std::filesystem::path p = base / "hello.txt";
    std::cout << "filename:  " << p.filename() << "\n";
    std::cout << "extension: " << p.extension() << "\n";
    std::cout << "parent:    " << p.parent_path() << "\n";

    // 查询文件
    std::cout << "exists:          " << std::filesystem::exists(p) << "\n";
    std::cout << "is_regular_file: " << std::filesystem::is_regular_file(p) << "\n";
    std::cout << "file_size:       " << std::filesystem::file_size(p) << "\n";

    // 递归遍历目录树
    std::cout << "entries:\n";
    for (const auto &entry : std::filesystem::recursive_directory_iterator(base))
        std::cout << "  " << entry.path() << "\n";

    // 复制后重命名
    std::filesystem::copy_file(p, base / "copy.txt");
    std::filesystem::rename(base / "copy.txt", base / "renamed.txt");

    // 清理
    std::filesystem::remove_all(base);
    std::cout << "after cleanup, exists: " << std::filesystem::exists(base) << "\n";
}

// -------------------------------------
int main(int argc, const char *argv[])
{
    // 设置控制台输入/输出代码页为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // 设置 stdout 为 UTF-8，避免 printf 走 ANSI 转换
    SetConsoleOutputCP(65001);

    std::cout << "-------------------------------------------------\n";
    func();
    std::cout << "-------------------------------------------------\n";

    //--------- 路径 std::filesystem::path -----------
    std::filesystem::path p = "/usr/local";
    std::print("the path p == {0}!\n", p.string());

    p /= "bin"; // 现在 p 为 /usr/local/bin
    std::print("the path p == {0}!\n", p.string());

    std::filesystem::path q = p / "clang"; // 拼接但不修改 p
    std::print("the path q == {0}!\n", p.string());
    std::print("the path p == {0}!\n", p.string());

    // path 提供了一组用于分解路径的成员函数
    std::filesystem::path path_ = "/usr/local/hello.txt";

    std::string filename_str = path_.filename().string();    // "hello.txt"
    std::string name_str     = path_.stem().string();        // "hello"
    std::string postfix_str  = path_.extension().string();   // ".txt"
    std::string folder_str   = path_.parent_path().string(); // "/usr/local"
    std::print("the filename of path == {0}!\n", filename_str);
    std::print("the name of path == {0}!\n", name_str);
    std::print("the postfix of path == {0}!\n", postfix_str);
    std::print("the folder of path == {0}!\n", folder_str);
    std::cout << "-------------------------------------------------\n";

    //--------- 查询文件状态 -----------
    std::error_code ec{};
    bool            flag{};
    flag = std::filesystem::exists(path_, ec); // 路径是否存在
    if (ec || !flag)
    {
        std::cout << "无法获取路径是否存在：" << ec.message() << std::endl;
    }
    flag = std::filesystem::is_regular_file(path_, ec); // 是否为常规文件
    if (ec || !flag)
    {
        std::cout << "无法获取是否为常规文件：" << ec.message() << std::endl;
    }
    flag = std::filesystem::is_directory(path_, ec); // 是否为目录
    if (ec || !flag)
    {
        std::cout << "无法获取是否为目录：" << ec.message() << std::endl;
    }
    flag = std::filesystem::file_size(path_, ec); // 文件大小（字节）
    if (ec || !flag)
    {
        std::cout << "无法获取大小：" << ec.message() << std::endl;
    }
    auto flag_time = std::filesystem::last_write_time(path_, ec); // 最后修改时间
    if (ec)
    {
        std::cout << "无法获取最后修改时间：" << ec.message() << std::endl;
    }
    std::cout << "-------------------------------------------------\n";

    //--------- 遍历目录 -----------
    auto dir = std::filesystem::path("./");
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        std::cout << entry.path() << std::endl;
    }

    // 递归遍历整棵目录树
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir))
    {
        if (entry.is_regular_file())
            std::cout << entry.path() << " (" << entry.file_size() << ")\n";
    }
    std::cout << "-------------------------------------------------\n";

    //--------- 创建、复制、删除 -----------
    // 递归创建目录（中间目录不存在时一并创建）
    flag = std::filesystem::create_directories(p / "a" / "b");

    // 复制单个文件
    auto src = std::filesystem::path("./src.cpp");
    auto dst = std::filesystem::path("../src.cpp");
    flag     = std::filesystem::copy_file(src, dst, ec);
    if (false == flag)
        std::cout << "复制文件失败: " << ec.message() << std::endl;

    // 递归复制目录
    std::filesystem::copy(src, dst, std::filesystem::copy_options::recursive);
    // 重命名 / 移动
    std::filesystem::rename("old_path", "new_path");
    // 删除单个文件或空目录
    std::filesystem::remove(p);
    // 递归删除，返回被删除的条目数
    std::filesystem::remove_all(p);

    return 0;
}

// cl /source-charset:utf-8 /execution-charset:utf-8
// clang++ 27_filesystem.cpp -std=c++23 -o demo.exe
// clang++ 27_filesystem.cpp -std=c++23 -finput-charset=UTF-8 -fexec-charset=UTF-8 -o demo.exe
