/**
 * @file 04_backtrace_file.cpp
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

#include <exception>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* Backtrace support 缓存异常日志
     * spdlog 提供了一个 backtrace 特性, 可以在日志记录过程中收集一定数量的日志消息
     * (即使这些消息没有被输出到日志文件, flush into disk file, just in buffer). 
     * 当出现某种关键事件时, 例如错误或异常, 开发者可以选择将收集到的 backtrace 输出,
     * 这样就能查看在事件发生之前发生了什么.
     */

    // 创建一个日志记录器并启用 backtrace 支持
    auto logger = spdlog::basic_logger_mt("basic_logger", "logs/backtrace_log.txt");
    // 启用 backtrace，并设置缓存 10 条日志
    logger->enable_backtrace(10);

    // 日志记录示例(这些日志将缓存到 backtrace 中, 而不是立即写入文件)
    for (size_t idx = 0; idx < 15; ++idx)
    {
        logger->debug("This is a debug message {}", idx);
    }

    // 模拟一个错误，并输出 backtrace 缓存的日志
    try
    {
        throw std::runtime_error("An error occurred!");
    }
    catch (const std::exception &e)
    {
        // 记录异常信息
        logger->error("Exception caught: {}", e.what());

        // 输出 backtrace 缓存的日志
        logger->dump_backtrace();
    }

    /* 代码解释:
     * 1. 启用 backtrace: 使用 logger->enable_backtrace(10); 
     *    来启用 backtrace 功能, 并指定最多缓存 10 条日志;
     * 2. 日志记录: 循环记录 15 条 debug 级别的日志信息; 这些日志不会立即写入文件, 而是先被缓存起来.
     * 3. 模拟错误: 在 try-catch 代码块中抛出一个异常, 并在 catch 块中捕获异常;
     * 
     * 输出 backtrace: 在捕获异常后, 使用 logger->dump_backtrace(); 
     *  将缓存的日志输出到日志文件, 这会帮助开发者查看在异常发生前, 程序经历了哪些函数调用和日志记录.
     * 
     * 输出结果: 在日志文件 backtrace_log.txt 中, 
     * 将看到异常发生之前的 10 条日志记录(最新的10条记录), 这些信息将帮助调试和追踪程序执行过程中的问题.
     * 
     * 使用场景:
     * 1. 错误调试: 当程序抛出异常或出现错误时, backtrace 能够帮助开发者查看程序执行的历史记录, 从而快速定位问题.
     * 2. 复杂系统: 在复杂的应用程序中, 程序可能会经过多个函数调用后才发生错误, 
     *    backtrace 提供了关键的上下文信息, 帮助开发者理解程序的执行路径.
     */

    return 0;
}
