/**
 * @file 02_daily_file.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/spdlog.h"

// ------------------------------------
int main(int argc, const char **argv)
{
    /* 3. 每日日志文件
     * spdlog 提供了 daily_file_sink 汇编器来创建每日日志文件.
     * daily_file_sink 每天都会创建一个名为 my_daily_log.txt 的新日志文件;
     * 日志消息将被写入此文件
     */
    // 创建名为 "daily_log" 并写入 "daily_log.txt" 的每日记录器
    auto daily_logger = spdlog::daily_logger_st("daily_log", "logs/daily_log.txt");

    // 将日志级别设置为调试
    daily_logger->set_level(spdlog::level::debug);

    // 记录一些消息
    daily_logger->info("这是一条信息消息");
    daily_logger->debug("这是一条调试消息");

    /* 还可以通过向 daily_file_sink 构造函数传递附加参数来自定义每日日志文件名和其他选项.
     * 例如 以下代码将创建名为 daily_log_%Y-%m-%d.txt 的每日日志文件:
     * "_st" ---> 表示单个线程;
     * "_mt" ---> 表示支持多线程安全;
     */
    // auto daily_logger_ = spdlog::daily_logger_st("daily_log_", "logs/daily_log.log");
    auto daily_logger_ = spdlog::daily_logger_mt("daily_log_", "logs/daily_log.log");
    daily_logger_->info("这是一条信息消息");
    daily_logger_->debug("这是一条调试消息");

    // Create a daily logger - a new file is created every day at 2:30 am
    auto logger = spdlog::daily_logger_mt("daily_logger", "logs/daily.txt", 2, 30);
    logger->set_level(spdlog::level::debug);
    logger->debug("This is Debug message log");
    logger->info("This is Debug message log");
    logger->warn("This is Debug message log");
    logger->error("This is Debug message log");
    logger->critical("This is Debug message log");

    return 0;
}
