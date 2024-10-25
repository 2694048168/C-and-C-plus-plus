/**
 * @file 11_synchrony_multi_sinks.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/async.h"                    // 支持异步日志
#include "spdlog/sinks/basic_file_sink.h"    // 文件输出 sink
#include "spdlog/sinks/stdout_color_sinks.h" // 控制台输出 sink
#include "spdlog/spdlog.h"

#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, const char **argv)
{
    /* 12. Asynchronous logger with multi sinks  带sink异步日志记录器
    * 在 spdlog 中, 可以创建一个异步日志记录器, 并将多个 sink(输出目标)组合在一起使用.
    * 这允许同时将日志输出到多个目标(例如文件、控制台等), 
    * 而且这些操作都是异步进行的, 不会阻塞主线程.
    *  
    */
    // 初始化异步日志线程池，队列大小为 8192，1 个后台线程
    spdlog::init_thread_pool(8192, 1);

    // 创建一个控制台 sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[console] [%^%l%$] %v");

    // 创建一个文件 sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/async_multi_sink_log.txt", true);
    file_sink->set_level(spdlog::level::debug);
    file_sink->set_pattern("[file] [%Y-%m-%d %H:%M:%S.%e] [%l] [thread %t] %v");

    // 创建一个异步日志记录器，包含多个 sinks
    auto async_logger
        = std::make_shared<spdlog::async_logger>("async_logger", spdlog::sinks_init_list({console_sink, file_sink}),
                                                 spdlog::thread_pool(), spdlog::async_overflow_policy::block);

    // 设置全局默认 logger（可选）
    spdlog::set_default_logger(async_logger);

    // 记录日志
    async_logger->info("This is an info message");
    async_logger->debug("This is a debug message");

    // 大量日志记录
    for (int i = 0; i < 10000; ++i)
    {
        async_logger->debug("Debug message number {}", i);
        async_logger->info("Info message number {}", i);
    }

    std::ofstream file_out;
    const char   *filepath = "logs/sample.txt";
    file_out.open(filepath);
    if (file_out.fail())
    {
        std::cout << "open the " << filepath << " failed\n";
    }

    for (size_t i = 0; i < 12; i++)
    {
        file_out << "the main-thread do something " << i << '\n';
    }
    file_out.close();

    // 确保所有日志都被写入
    spdlog::shutdown();

    /* 代码解释:
    * 1. 初始化线程池: spdlog::init_thread_pool(8192, 1); 
    * ----初始化了一个大小为 8192 的队列和 1 个处理线程的线程池, 专用于异步日志记录;
    * 2. 创建多个 sinks:
    * ----控制台 sink: stdout_color_sink_mt 用于将日志输出到控制台, 
    *     并设置了日志级别为 info, 日志格式带有自定义的控制台前缀.
    * ----文件 sink: basic_file_sink_mt 用于将日志记录到文件 async_multi_sink_log.txt 中,
    *    日志级别为 debug, 并且带有时间戳、线程 ID 等详细信息.
    * 3. 创建异步日志记录器: 使用 spdlog::async_logger 创建一个异步日志记录器 async_logger,
    *   将控制台 sink 和文件 sink 组合在一起, 所有日志操作都在后台线程中处理, 不会阻塞主线程.
    * ----spdlog::async_overflow_policy::block 指定当日志队列满时, 主线程将阻塞, 等待队列腾出空间;
    * 4. 记录日志: 通过 async_logger->info() 和 async_logger->debug() 记录了几条日志消息.
    * ----循环记录大量日志消息以测试异步日志记录的性能.
    * 5. 日志刷新与关闭: 在程序结束前, 调用 spdlog::shutdown(); 确保所有日志消息都被处理并写入到各个目标.
    * 
    * 注意事项:
    * 1. 队列大小: 异步日志记录器依赖队列来管理日志消息的处理, 
    *    队列大小的设置需要根据日志量和性能要求进行权衡.
    * 2. 性能优化: 对于高性能需求的场景, 合理配置线程池和队列可以最大化性能, 同时避免队列溢出或主线程阻塞.
    * 
    */

    return 0;
}
