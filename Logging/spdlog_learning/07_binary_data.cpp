/**
 * @file 07_binary_data.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/fmt/bin_to_hex.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <array>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* Log binary data in hex 记录二进制
     * spdlog 提供了一种方便的方法来记录二进制数据, 并将其以十六进制格式输出.
     * 这个特性特别有用, 当你需要调试或分析底层数据时;
     * 例如网络数据包、文件内容或硬件接口数据.
     * 在 spdlog 中, 这个功能是通过 fmt/bin_to_hex.h 实现的,
     * 它允许开发者将各种类型的二进制数据转换为十六进制格式, 并灵活地控制输出格式.
     * 
     * 特性介绍
     * 1. 容器支持: spdlog 的 to_hex 函数支持多种标准容器,
     *    如 std::vector<char>, std::array<char>, std::string 等, 甚至支持原生指针或迭代器范围.
     * 2. 格式标志: 通过格式标志,开发者可以控制十六进制输出的样式,
     *    例如是否大写、是否显示字节分隔符、是否显示 ASCII 字符等.
     ? 支持的格式标志:
     * 1. {:X}: 以大写字母显示十六进制数据(例如 A-F);
     * 2. {:s}: 不使用空格分隔每个字节;
     * 3. {:p}: 不在每行开头显示位置信息(偏移量);
     * 4. {:n}: 不将输出拆分为多行;
     * 5. {:a}: 如果未设置 :n 标志, 显示 ASCII 表示形式;
     * 
     */
    // 初始化控制台日志记录器
    spdlog::set_level(spdlog::level::info); // 设置日志级别
    spdlog::set_default_logger(spdlog::stdout_color_mt("console"));

    // 获取一个控制台日志记录器
    auto console = spdlog::get("console");

    // 创建一个包含二进制数据的缓冲区
    std::array<char, 80> buf = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'S', 'p', 'd', 'l', 'o', 'g', '!'};

    // 以默认格式记录二进制数据
    console->info("Binary example: {}", spdlog::to_hex(buf));

    // 仅记录前10个字节，并且不换行
    console->info("Another binary example: {:n}", spdlog::to_hex(std::begin(buf), std::begin(buf) + 10));

    // 更多格式示例:
    // 大写输出
    console->info("Uppercase: {:X}", spdlog::to_hex(buf));

    // 大写输出，且不使用分隔符
    console->info("Uppercase, no delimiters: {:Xs}", spdlog::to_hex(buf));

    // 大写输出，不使用分隔符，且不显示位置信息
    console->info("Uppercase, no delimiters, no position info: {:Xsp}", spdlog::to_hex(buf));

    // 默认格式，并附带 ASCII 显示
    console->info("With ASCII: {}", spdlog::to_hex(buf));

    /* 代码解释
     *1. 包含必要的头文件: spdlog/spdlog.h 是日志功能的核心头文件,
     *   spdlog/fmt/bin_to_hex.h 提供了将二进制数据转换为十六进制的功能.
     *2. 创建并填充缓冲区: std::array<char, 80> 是一个包含二进制数据的缓冲区, 示例中填入了一些字符数据.
     *3. 记录二进制数据: 默认记录二进制数据为十六进制格式;
     *4. 使用不同的格式标志控制输出样式, 如是否大写、是否显示分隔符和偏移量等;
     *5. 初始化日志记录器: 使用 spdlog::stdout_color_mt("console") 
     *   初始化控制台日志记录器,并设置日志级别.
     */

    return 0;
}
