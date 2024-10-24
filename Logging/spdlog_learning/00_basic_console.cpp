/**
 * @file basic_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <memory>

// -----------------------------------
int main(int argc, const char **argv)
{
    /** 1. 标准输出(stdout)和标准错误(stderr)的日志
     * 标准输出(stdout): 通常用于输出普通信息, 如程序的正常运行状态等.
     * 标准错误(stderr): 通常用于输出错误信息或警告, 以便于在大量日志信息中突出显示这些重要信息.
     * spdlog 提供了多种日志 sink(即日志输出目标)来实现上述功能.
     * 可以创建一个日志器, 将多个 sink 组合在一起, 使其同时记录到 stdout 和 stderr.
     */

    // 创建 stdout 和 stderr sinks
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();

    // 创建一个 logger，组合这两个 sinks
    spdlog::logger logger("multi_sink", {stdout_sink, stderr_sink});

    // 设置日志级别（可选）
    logger.set_level(spdlog::level::debug);

    // 使用 logger 记录日志
    logger.info("This is an info message");
    logger.warn("This is a warning message");
    logger.error("This is an error message");

    /* 解释
     * 标准输出(stdout): 通常用于输出普通信息,如程序的正常运行状态等;
     * 标准错误(stderr): 通常用于输出错误信息或警告,以便于在大量日志信息中突出显示这些重要信息;
     * spdlog 提供了多种日志 sink(即日志输出目标)来实现上述功能,
     * 可以创建一个日志器, 将多个 sink 组合在一起，使其同时记录到 stdout 和 stderr.
     * 
     * spdlog 中的 sinks: 提供了一系列预定义的 sinks, 供用户选择和组合, 以满足不同的日志记录需求;
     * 每个 sink 都有不同的功能和特性, 用户可以根据需要将多个 sinks 组合在一起, 创建复杂的日志记录系统;
     * 常用的 spdlog sinks: 
     * 1. stdout_sink: 将日志输出到标准输出(即控制台);
     * 2. stderr_sink: 将日志输出到标准错误;
     * 3. basic_file_sink: 将日志输出到一个基础文件中;
     * 4. rotating_file_sink: 将日志输出到一个文件中, 并根据文件大小进行轮转;
     * 5. daily_file_sink: 将日志输出到一个文件中, 并根据日期进行轮转;
     * 6. null_sink: 将日志丢弃, 相当于禁用输出;
     * 7. syslog_sink: 将日志输出到系统日志(如 Linux 的 syslog);
     * 8. tcp_sink: 将日志输出到 TCP 网络连接;
     * 9. udp_sink: 将日志输出到 UDP 网络连接;
     * 
     * 在 spdlog 中 sinks 是日志输出的目标, 可以将日志记录到控制台、文件或网络等不同的地方;
     * 通过组合多个 sinks, 用户可以灵活地配置日志记录系统, 以满足复杂的日志记录需求.
     * 创建一个 rotating file sink 的日志器 
     */
    // 创建一个 rotating file sink
    auto rotating_sink
        = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/rotating_log.txt", 1024 * 1024 * 5, 3);

    // 创建一个 logger，使用 rotating file sink
    spdlog::logger logger_("file_logger", {stdout_sink, rotating_sink});

    // 设置日志级别（可选）
    logger_.set_level(spdlog::level::info);

    // 使用 logger 记录日志
    for (int idx = {0}; idx < 10; ++idx)
    {
        logger_.info("This is an info message in a rotating log file");
    }

    return 0;
}
