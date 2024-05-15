/**
 * @file 00_filesystemConcepts.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <filesystem>
#include <iostream>

/**
 * @brief Filesystem Concepts 文件系统的相关概念
 * 如何使用 stdlib 提供的文件系统库来进行文件系统的相关操作,
 * 例如检查和操作文件, 枚举目录, 以及与文件流的互操作.
 * 
 * ====文件系统有几个重要的概念, 其中核心的是文件.
 * 文件是支持输出和输入并保存数据的文件系统对象, 文件存在于目录中, 目录又可以进行嵌套;
 * 在文件系统中, 为了简单起见, 把目录也视为文件, 包含文件的目录称为该文件的父目录;
 * 路径是标识具体文件的字符串, 路径都以特定的根字符串表示,
 * 例如在 Windows 上以C: 或者 //localhost 标识根路径;
 * 在 Unix 上以 / 标识根路径;
 * 路径的其他部分由分隔符分开, 路径终止于非目录文件;
 * 路径可以包含"."和"..", 用来表示当前目录和父目录;
 * 
 * ====硬链接(hard link)为已经存在的目录分配别名;
 * 符号链接(symbolic link)为路径(可能存在也可能不存在)分配一个别名;
 * 
 * ====
 * 1. 相对路径[relative path]: 相对于另一条路径(通常是当前目录)指定位置的路径称为相对路径;
 * 2. 规范路径[ canonical path]: 可以明确表示一个文件的位置, 不包含"."和"..", 并且不包含符号链接;
 * 3. 绝对路径[absolute path]: 是明确标识文件位置的路径, 规范路径和绝对路径的主要区别是规范路径不能包含"."和"..";
 * 
 * TODO:The stdlib filesystem might not be available 
 * ?if the target platform doesn’t offer a hierarchical filesystem.
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief std::filesystem::path 是文件系统库中用于构造路径的类,
     *  有很多方法可以构建路径; 最常见的两个方法是使用默认构造函数(构造空的路径对象)
     *  和接受字符串参数的构造函数(构造一个字符串指向的路径).
     * *＜filesystem＞ 头文件 std=c++17
     * 
     * 如何从字符串构造路径对象, 以及如何将其分解为各个组成部分, 对其进行修改;
     * 在许多常见的系统编程和应用编程中, 都需要与文件系统交互.
     * 因为每个操作系统都有一个特有的文件系统, stdlib 的文件系统是对各个操作系统文件系统的抽象,
     * 使得可以编写跨平台的代码.
     * 
     */
    std::cout << "[====]std::filesystem::path supports == and .empty()\n";
    std::filesystem::path empty_path;
    std::filesystem::path shadow_path{"/etc/shadow"};
    if (empty_path.empty())
        std::cout << "the empty_path is empty: " << empty_path << '\n';
    assert(shadow_path == "/etc/shadow");

    /**
     * @brief 分解路径 Decomposing Paths
     * ?path 类包含一些分解方法, 它们实际上是专门的字符串操纵符, 允许提取路径的各个部分
     * 1. root_name() 返回根名字;
     * 2. root_directory() 返回根目录;
     * 3. root_path() 返回根路径;
     * 4. relative_path() 返回相对于根的路径;
     * 5. parent_path() 返回父路径;
     * 6. filename() 返回文件名;
     * 7. stem() 返回去掉扩展名的文件名;
     * 8. extension() 返回扩展名;
     * 
     * 没有一种分解方法要求路径实际指向现有文件, 只需提取路径内容的组成部分, 而不是指向的文件
     * 分隔符方面: Linux 使用正斜杠'/', Windows 使用正斜杠'\\', 使用原生字符串
     * 
     * TODO:对于 Windows 的一个非常重要的系统库 kernel32.dll, 以上每个方法都打印了关于这个系统库的某值
     */
    const std::filesystem::path kernel32{R"(C:\Windows\System32\kernel32.dll)"};
    std::cout << "\nRoot name: " << kernel32.root_name();
    std::cout << "\nRoot directory: " << kernel32.root_directory();
    std::cout << "\nRoot path: " << kernel32.root_path();
    std::cout << "\nRelative path: " << kernel32.relative_path();
    std::cout << "\nParent path: " << kernel32.parent_path();
    std::cout << "\nFilename: " << kernel32.filename();
    std::cout << "\nStem: " << kernel32.stem();
    std::cout << "\nExtension: " << kernel32.extension();

    /**
     * @brief 修改路径 Modifying Paths
     * ?path 还提供了几种修改方法,允许修改路径:
     * 1. clear() 清空路径;
     * 2. make_preferred() 将所有目录分隔符转换为实现首选的目录分隔符; 
     *    在Windows 上, 这会将通用分隔符 / 转换为系统首选的分隔符 \;
     * 3. remove_filename() 删除路径的文件名部分;
     * 4. replace_filename(p) 将路径的文件名替换为另一个路径 p 的文件名;
     * 5. replace_extension(p) 用另一个路径 p 的扩展名替换路径的扩展名;
     * 6. remove_extension() 删除路径的扩展名部分;
     * 
     */
    std::filesystem::path path{R"(C:/Windows/System32/kernel32.dll)"};
    std::cout << path << std::endl;

    path.make_preferred();
    std::cout << path << std::endl;

    path.replace_filename("win32k-full.sys");
    std::cout << path << std::endl;

    path.remove_filename();
    std::cout << path << std::endl;

    path.clear();
    std::cout << "Is empty: " << std::boolalpha << path.empty() << std::endl;

    // 文件系统路径的方法
    // TODO: https://en.cppreference.com/w/cpp/filesystem/path

    return 0;
}
