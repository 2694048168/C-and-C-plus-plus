/**
 * @file 01_basic_file.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

// ------------------------------------
int main(int argc, const char **argv)
{
    /* 2. 基本的文件日志器(basic file logger)
     * 步骤
     * 包含头文件: 需要包含 spdlog 的核心头文件以及用于文件日志器的头文件;
     * 创建日志器: 使用 basic_file_sink_mt 创建一个文件日志器;
     * 记录日志: 使用日志器记录不同级别的日志信息;
     */
    // 创建一个文件日志器
    auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic_log.txt");

    // 设置日志级别（可选）
    file_logger->set_level(spdlog::level::debug);

    // 设置日志格式（可选）
    file_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

    // 使用日志器记录日志 不断追加方式append
    file_logger->debug("This is a debug message");
    file_logger->info("This is an info message");
    file_logger->warn("This is a warning message");
    file_logger->error("This is an error message");
    file_logger->critical("This is a critical message");

    return 0;
}
