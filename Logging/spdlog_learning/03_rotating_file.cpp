/**
 * @file 03_rotating_file.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 4. Rotating files 
     * Rotating Files(轮转文件): 指的是日志文件达到一定大小或者数量后,
     * 旧的日志文件会被重命名保存, 新的日志内容会写入到一个新的文件中;
     * 这种机制有助于控制日志文件的大小和数量, 从而避免磁盘被大量日志文件占满.
     */
    // Create a file rotating logger with 5 MB size max and 3 rotated files
    // 5 MB = 5 * 1024 KB = 5 * 1024 * 1024 Byte
    // auto max_size  = 1024 * 1024 * 5;
    auto max_size  = 1024;
    auto max_files = 3;

    // 工作原理: 在 logs/ 目录下, rotating.log 将成为主要日志文件;
    // 当 rotating.log 的大小达到 5MB 时, spdlog 会将它重命名为 rotating.1.log,
    // 新的日志内容将写入新的 rotating.log; 这个过程会继续进行, 直到有 3 个日志文件
    // ?(rotating.log, rotating.1.log, rotating.2.log), 最旧的日志文件会被删除或覆盖.
    auto logger = spdlog::rotating_logger_mt("rotating_logger", "logs/rotating.log", max_size, max_files);

    // 写日志
    for (size_t idx{0}; idx < 100; ++idx)
    {
        logger->info("This is an info message");
    }

    // 刷新日志(将缓存内容写入文件)
    spdlog::flush_on(spdlog::level::info);

    return 0;
}
