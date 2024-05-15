/**
 * @file 02_inspectingFiles.cpp
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
 * @brief Inspecting Files and Directories 检查文件和目录
 * ?可以使用以下函数检查文件系统的各种属性:
 * 1. current_path([p], [ec])，如果提供了p, 则将程序的当前路径设置为 p;否则, 它返回程序的当前路径;
 * 2. exists(p, [ec]) 返回 p 处是否存在文件或目录;
 * 3. equivalent(p1, p2, [ec]) 返回 p1 和 p2 是否引用相同的文件或目录;
 * 4. file_size(p, [ec]) 返回 p 处常规文件的大小(以字节为单位);
 * 5. hard_link_count(p, [ec]) 返回 p 的硬链接数;
 * 6. last_write_time(p, [t] [ec]), 如果提供了t, 则将 p 的最后修改时间设置为 t;
 *    否则它返回上次修改 p 的时间(t 是一个 std::chrono::time_point);
 * 7. permissions(p, prm, [ec]) 设置 p 的权限, prm 的类型是 std::filesystem::perms,
 *    它是一个模仿 POSIX 权限位的枚举类;
 * 8. read_symlink(p, [ec]) 返回符号链接 p 的目标;
 * 9. space(p, [ec]) 以 std::filesystem::space_info 的形式返回有关文件系统 p 占用的空间信息,
 *    此 POD 包含三个字段: capacity(总大小), free(可用空间)和 available(非特权进程可用的空间).
 *     它们都是无符号整数类型, 以字节为单位;
 * 10. status(p, [ec]) 以 std::filesystem::file_status 的形式返回文件或目录 p 的类型和属性,
 *     该类包含一个不接受参数并返回 std::filesystem::file_type 类型的对象的 type 方法,
 *     该对象是一个枚举类, 它接受描述文件类型的值,  例如 not_found、regular、directory.
 *     符号链接 file_status 类 还提供了一个不接受任何参数并返回 std::filesystem::perms 类型的对象的permissions 方法.
 * 11. symlink_status(p, [ec]) 是不跟随符号链接的文件系统状态;
 * 
 */

void write_info(const std::filesystem::path &p)
{
    if (!exists(p))
    {
        std::cout << p << " does not exist." << std::endl;
        return;
    }
    const auto last_write = last_write_time(p).time_since_epoch();

    const auto in_hours = duration_cast<std::chrono::hours>(last_write).count();

    std::cout << p << "\t" << in_hours << "\t" << file_size(p) << "\n";
}

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 操作文件和目录 Manipulating Files and Directories
     * ?文件系统库包含许多用于操作文件和目录的方法:
     * 1. copy(p1, p2, [opt], [ec]) 将文件或目录从 p1 复制到 p2,
     *    可以提供std::filesystem::copy_options opt 来自定义 copy_file 的行为,
     *   这个枚举类可以取多个值, 包括 none(如果目标已经存在,则报告错误),
     *   skip_existing(保持现有的), overwrite_existing(覆盖)和 update_existing(如果 p1 较新,则覆盖);
     * 2. copy_file(p1, p2, [opt], [ec]) 类似于 copy, 但如果 p1 不是常规文件,它将生成错误;
     * 3. create_directory(p, [ec]) 创建目录 p;
     * 4. create_directories(p, [ec]) 就像递归调用 create_directory 一样,
     *    所以如果嵌套路径包含不存在的父级, 请使用此形式;
     * 5. create_hard_link(tgt, lnk, [ec]) 在 lnk 创建到 tgt 的硬链接;
     * 6. create_symlink(tgt, lnk, [ec]) 在 lnk 创建到 tgt 的符号链接;
     * 7. create_directory_symlink(tgt, lnk, [ec]) 应该用于目录, 与 create_symlink 不同;
     * 8. remove(p, [ec]) 删除文件或空目录 p(不跟随符号链接);
     * 9. remove_all(p, [ec]) 递归地删除文件或目录 p(不跟随符号链接);
     * 10. rename(p1, p2, [ec]) 将 p1 重命名为 p2;
     * 11. resize_file(p, new_size, [ec]) 将 p 的大小(如果它是常规文件)更改为new_size,
     *    如果此操作增大文件, 则用零填充新空间;否则, 该操作从末尾开始修剪 p;
     * 
     */
    const std::filesystem::path win_path{R"(C:/Windows/System32/kernel32.dll)"};
    const auto                  readme_path = std::filesystem::temp_directory_path() / "README";

    try
    {
        write_info(win_path);
        write_info(readme_path);

        std::cout << "Copying " << win_path.filename() << " to " << readme_path.filename() << "\n";
        copy_file(win_path, readme_path);
        write_info(readme_path);

        std::cout << "Resizing " << readme_path.filename() << "\n";
        resize_file(readme_path, 1024);
        write_info(readme_path);

        std::cout << "Removing " << readme_path.filename() << "\n";
        remove(readme_path);
        write_info(readme_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
