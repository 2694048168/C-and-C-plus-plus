/**
 * @file 10_synchrony_log.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/async.h"                 // 支持异步日志
#include "spdlog/sinks/basic_file_sink.h" // 支持文件输出
#include "spdlog/spdlog.h"

#include <iostream>
#include <memory>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 11. Asynchronous logging 异步日志记录
    * spdlog 支持异步日志记录, 这意味着日志消息可以在一个独立的线程中被处理和写入日志目标,
    * 而不会阻塞主线程的执行; 这对于高性能应用程序尤为重要, 因为它可以显著减少因日志记录导致的性能开销.
    * 
    * 异步日志记录的特性:
    * 1. 非阻塞: 日志记录操作在独立线程中异步处理, 主线程无需等待日志写入完成;
    * 2. 高效: 适用于高吞吐量的日志记录需求, 尤其在日志量大且对性能敏感的场景下;
    * 3. 队列管理: spdlog 通过内部的队列管理日志消息, 可以控制队列的大小及溢出策略;
    * 
    * // default thread pool settings can be modified *before* creating the async logger:
    * // spdlog::init_thread_pool(8192, 1); // queue with 8k items and 1 backing thread.
    * auto async_file = spdlog::basic_logger_mt<spdlog::async_factory>("async_file_logger", "logs/async_log.txt");
    * // alternatively:
    * // auto async_file = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>("async_file_logger", "logs/async_log.txt");   
    * 
    */
    // 设置异步日志模式，并指定队列大小为 8192 条日志消息
    spdlog::init_thread_pool(8192, 1);

    // 创建一个异步文件 sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/async_log.txt", true);

    // 创建一个异步日志记录器
    auto async_logger = std::make_shared<spdlog::async_logger>("async_logger", file_sink, spdlog::thread_pool(),
                                                               spdlog::async_overflow_policy::block);

    // 设置异步日志记录器为全局默认日志记录器（可选）
    spdlog::set_default_logger(async_logger);

    // 使用异步日志记录器记录一些日志消息
    for (int i = 0; i < 10000; ++i)
    {
        async_logger->info("This is async log message number {}", i);
    }

    for (size_t i = 0; i < 12; i++)
    {
        std::cout << "the main-thread do something " << i << '\n';
    }

    // 刷新日志，确保所有消息都被写入文件
    spdlog::shutdown();

    /* 代码解释
    * 1. 初始化线程池:
    * ----spdlog::init_thread_pool(8192, 1); 初始化一个线程池用于异步日志记录;
    *   第一个参数 8192 表示队列的最大消息数; 第二个参数 1 表示线程池中线程的数量.
    * 2. 创建异步 sink:
    * ----std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/async_log.txt", true); 
    *   创建了一个文件 sink, 用于将日志记录到文件 async_log.txt 中;
    *   true 参数表示文件为追加模式, 即新的日志将被追加到已有文件的末尾;
    * 3. 创建异步 logger:
    * ----spdlog::async_logger 创建一个异步日志记录器;
    * ----spdlog::thread_pool() 获取之前初始化的线程池, 用于处理异步日志;
    * ----spdlog::async_overflow_policy::block 指定当队列满时, 主线程将阻塞等待队列有空余位置;
    * 4. 设置默认日志记录器(可选): 使用 spdlog::set_default_logger(async_logger); 
    * ----将这个异步日志记录器设置为默认的日志记录器, 这样在程序的其他地方
    *     可以直接使用 spdlog::info 等方法进行日志记录.
    * 5. 记录日志: 通过 for 循环记录大量日志消息, 因为是异步日志记录, 
    *    主线程不会因日志写入而被阻塞, 日志会被异步写入文件.
    * 6. 刷新和关闭: 在程序结束前, 调用 spdlog::shutdown(); 以确保所有日志消息都被处理并写入到文件.
    *
    * 注意事项:
    * 1. 队列大小: 队列大小的设置需要根据应用的日志量和处理能力进行调整;
    *    如果队列过小, 可能会导致日志消息溢出或主线程阻塞;
    * 2. 异步日志的潜在延迟: 由于日志是在后台线程中处理的, 日志消息可能不会立即写入目标,
    *    对于需要即时记录的关键日志, 可能需要考虑同步日志记录或手动刷新.
    * 
    */

    return 0;
}
