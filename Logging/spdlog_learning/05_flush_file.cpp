/**
 * @file 05_flush_file.cpp
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

#include <chrono>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 6. Periodic flush 定期刷新日志缓冲区
     * Periodic flush 是 spdlog 提供的一种功能, 用于定期刷新日志缓冲区.
     * 默认情况下, spdlog 会将日志消息保存在内存缓冲区中, 并在需要时(如日志缓冲区满或手动调用 flush 函数)
     * 将这些消息写入目标日志文件或输出流.
     * 使用 Periodic flush 功能, 开发者可以设置一个固定的时间间隔, 
     * 让 spdlog 自动定期刷新日志缓冲区, 将消息写入磁盘或输出流.
     * 这对于确保日志数据不会因应用程序异常终止而丢失非常有用.
     * 
     * 使用场景:
     * 1. 可靠性: 在长时间运行的应用程序中, 定期刷新日志可以防止日志消息因为应用崩溃或异常终止而丢失.
     * 2. 实时性: 定期刷新可以确保日志更接近实时反映程序状态, 特别是在调试和监控应用程序时.
     * 
     */

    // 创建一个基本的文件日志记录器
    auto logger = spdlog::basic_logger_mt("periodic_logger", "logs/periodic_flush_log.txt");
    // 设置日志记录器的刷新间隔为3秒
    spdlog::flush_every(std::chrono::seconds(3));

    // 写入一些日志消息
    for (size_t idx{0}; idx < 100; ++idx)
    {
        logger->info("Logging message number {}", idx);
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 模拟一些处理延迟
    }

    // 结束前手动刷新一次
    spdlog::shutdown();

    /* 代码解释:
    * 1. 创建日志记录器: 使用 spdlog::basic_logger_mt 创建一个记录日志到文件的日志记录器,
    *    日志文件名为 periodic_flush_log.txt;
    * 2. 设置定期刷新间隔: 使用 spdlog::flush_every 设置日志记录器每 3 秒刷新一次缓冲区,
    *    将日志消息写入文件;
    * 3. 日志记录: 在一个循环中写入 100 条日志消息, 每条日志消息之间有 1 秒的延迟;
    * 4. 手动刷新并关闭: 在程序结束前, 使用 spdlog::shutdown() 手动刷新一次
    *    所有日志记录器并关闭它们, 以确保所有日志都被写入文件;
    * 
    * 注意事项:
    * 1. 性能权衡: 定期刷新虽然能提高日志的实时性和可靠性, 但频繁的刷新操作可能会影响性能,
    *    特别是在高频率日志记录的情况下; 开发者需要根据应用程序的实际需求权衡刷新频率与性能之间的关系.
    * 2. 多日志记录器环境: 在一个程序中可能会有多个日志记录器, flush_every 适用于所有日志记录器,
    *    而不仅仅是某一个特定的日志记录器.
    */

    return 0;
}
