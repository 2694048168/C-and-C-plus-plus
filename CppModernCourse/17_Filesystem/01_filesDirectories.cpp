/**
 * @file 01_filesDirectories.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <filesystem>
#include <iostream>

/**
 * @brief Files and Directories 文件和目录
 * ?std::filesystem::path 类是文件系统库的核心, 但实际上它的方法都不与文件系统交互.
 * 但是＜filesystem＞ 头文件中包含了一些非成员函数, 它们可以做这些事情.
 * 将 std::filesystem::path 对象视为一种声明想与哪种文件系统组件进行交互的方式,
 * 将 ＜filesystem＞ 看作包含对这些组件执行操作的函数的头文件, 这些函数具有友好的错误处理接口,
 * 并允许将路径分解为目录名、文件名和扩展名等, 基于这些函数可以使用许多工具与环境中的文件进行交互,
 * *而无须使用特定于操作系统的应用程序编程接口.
 * 
 * ====Error Handling 错误处理
 * ?与文件系统交互涉及潜在的错误, 例如找不到文件、权限不足或不支持操作.
 * 文件系统库<filesystem>中与文件系统交互的每个非成员函数都必须向调用者传达错误条件,
 * *这些非成员函数提供了两个选项: 抛出异常或设置错误变量;
 * 每个函数都有两个重载: 一个允许传递对 std::system_error 的引用, 另一个忽略此参数;
 * 如果提供了引用, 该函数将 system_error 设置为一个错误条件;
 * 如果不提供此引用, 则该函数将抛出 std::filesystem::filesystem_error(继承自std::system_error 的异常类型).
 * 
 */

void describe(const std::filesystem::path &p)
{
    std::cout << std::boolalpha << "Path: " << p << std::endl;
    try
    {
        std::cout << "Is directory: " << std::filesystem::is_directory(p) << std::endl;
        std::cout << "Is regular file: " << std::filesystem::is_regular_file(p) << std::endl;
    }
    catch (const std::exception &exp)
    {
        std::cerr << "Exception: " << exp.what() << std::endl;
    }
}

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 构造路径的函数 Path-Composing Functions
     * 作为使用 path 的构造函数的替代方法, 可以使用以下函数构造各种路径:
     * 1. absolute(p, [ec]) 返回引用与 p 相同位置但 is_absolute() 为 true 的绝对路径;
     * 2. canonical(p, [ec]) 返回引用与 p 相同位置的规范路径;
     * 3. current_path([ec]) 返回当前路径;
     * 4. relative(p, [base], [ec]) 返回 p 相对于 base 的路径;
     * 5. temp_directory_path([ec]) 返回临时文件的目录, 结果保证是已经存在的目录;
     * 
     */
    try
    {
        const auto temp_path = std::filesystem::temp_directory_path();
        const auto relative  = std::filesystem::relative(temp_path);

        std::cout << std::boolalpha << "Temporary directory path: " << temp_path;
        std::cout << "\nTemporary directory absolute: " << temp_path.is_absolute();
        std::cout << "\nCurrent path: " << std::filesystem::current_path();
        std::cout << "\nTemporary directory's relative path: " << relative;
        std::cout << "\nRelative directory absolute: " << relative.is_absolute();
        std::cout << "\nChanging current directory to temp.";

        std::filesystem::current_path(temp_path);
        std::cout << "\nCurrent directory: " << std::filesystem::current_path();
    }
    catch (const std::exception &exp)
    {
        // 如果系统不支持某些操作, 甚至可能会抛出异常
        // ?警告: C++ 标准允许某些环境可能不支持部分或全部文件系统库
        std::cerr << "Error: " << exp.what();
    }

    /**
     * @brief 检查文件类型 Inspecting File Types
     * ?可以使用以下函数检查给定路径的文件属性:
     * 1. is_block_file(p, [ec]) 确定p是否为块文件,即某些操作系统中的特殊文件
     *    (例如, Linux 中的块设备允许以固定大小的块传输随机可访问的数据);
     * 2. is_character_file(p, [ec]) 确定p是否为字符文件,即某些操作系统中的特殊文件
     *    (例如, Linux 中允许发送和接收单个字符的字符设备);
     * 3. is_regular_file(p, [ec]) 确定 p 是否为常规文件;
     * 4. is_symlink(p, [ec]) 确定 p 是否为符号链接, 即是否为对另一个文件或目录的引用;
     * 5. is_empty(p, [ec]) 确定 p 是空文件还是空目录;
     * 6. is_directory(p, [ec]) 确定 p 是否为目录;
     * 7. is_fifo(p, [ec]) 确定 p 是否为命名管道, 即很多操作系统都支持的进程间通信机制;
     * 8. is_socket(p, [ec]) 确定 p 是否为套接字, 即许多操作系统都支持的另一种特殊的进程间通信机制;
     * 9. is_other(p, [ec]) 确定 p 是否为常规文件、目录或符号链接之外的某种文件;
     * 
     */
    std::filesystem::path win_path{R"(C:/Windows/System32/kernel32.dll)"};

    describe(win_path);
    win_path.remove_filename();
    describe(win_path);

    std::filesystem::path nix_path{R"(/bin/bash)"};

    describe(nix_path);
    nix_path.remove_filename();
    describe(nix_path);

    return 0;
}
