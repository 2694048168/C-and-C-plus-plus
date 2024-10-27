/**
 * @file 17_command_line.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/cfg/argv.h" // 支持从命令行参数加载日志级别
#include "spdlog/cfg/env.h"  // 支持从环境变量加载日志级别
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

int main(int argc, const char **argv)
{
    /* 18.Load log levels from the env variable or argv
    * 在 spdlog 中, 可以通过环境变量或命令行参数动态设置日志级别.
    * 这种方式非常灵活, 尤其是在运行时环境不可控或需要根据不同的部署环境动态调整日志级别时.
    * 这种功能对于调试、测试和生产环境中的日志管理非常有用.
    */

    // 从环境变量加载日志级别
    spdlog::cfg::load_env_levels();

    // 或者，从命令行参数加载日志级别
    // 例如: ./example SPDLOG_LEVEL=info,my_logger=trace
    spdlog::cfg::load_argv_levels(argc, argv);

    // 创建一个日志记录器并记录消息
    auto logger = spdlog::stdout_color_mt("my_logger");
    logger->trace("This is a trace message");
    logger->debug("This is a debug message");
    logger->info("This is an info message");
    logger->warn("This is a warning message");
    logger->error("This is an error message");
    logger->critical("This is a critical message");

    // 创建一个文件日志器
    auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic_log.txt");
    // 使用日志器记录日志 不断追加方式append
    file_logger->trace("This is a trace message");
    file_logger->debug("This is a debug message");
    file_logger->info("This is an info message");
    file_logger->warn("This is a warning message");
    file_logger->error("This is an error message");
    file_logger->critical("This is a critical message");

    /* 代码解释:
    * 1. 加载日志级别:
    * ---从环境变量加载:
    *   --使用 spdlog::cfg::load_env_levels(); 从环境变量 SPDLOG_LEVEL 中加载日志级别;
    *   --例如设置环境变量 export SPDLOG_LEVEL=info,my_logger=trace,
    *     可以控制所有日志记录器和名为 my_logger 的日志记录器的日志级别.
    * ---从命令行参数加载:
    *   --使用 spdlog::cfg::load_argv_levels(argc, argv); 从命令行参数中加载日志级别;
    *   --例如通过命令 ./example SPDLOG_LEVEL=info,my_logger=trace 运行程序, 可以设置日志级别;
    * 2. 创建日志记录器并记录日志:
    * ---使用 spdlog::stdout_color_mt("my_logger"); 创建一个带有彩色控制台输出的日志记录器.
    * ---调用不同级别的日志记录方法（如 debug, info, warn, error）来记录日志消息.
    * 
    * 3. 使用示例: 从环境变量加载日志级别, 首先设置环境变量[export SPDLOG_LEVEL=info,my_logger=trace]
    * 在这种情况下, SPDLOG_LEVEL=info 设置所有日志记录器的默认级别为 info;
    * my_logger=trace 设置名为 my_logger 的日志记录器的级别为 trace;
    * --从命令行参数加载日志级别,运行程序时传递日志级别参数:
    * ./example SPDLOG_LEVEL=warn,my_logger=debug
    * 在这种情况下, SPDLOG_LEVEL=warn 设置所有日志记录器的默认级别为 warn;
    * my_logger=debug 设置名为 my_logger 的日志记录器的级别为 debug.
    * 
    * 可能的输出,根据设置的日志级别,不同级别的日志消息可能会被显示或忽略.
    * 例如如果日志级别设置为 info,则 debug 级别的消息不会被输出.
    * 
    * 总结: 通过使用 spdlog::cfg::load_env_levels() 或 spdlog::cfg::load_argv_levels()
    * 可以在运行时灵活地控制日志级别, 这为日志管理提供了极大的便利,
    * 使得在不同环境下调试和运行程序变得更加简单和高效.
    * 无论是通过环境变量还是命令行参数, 都可以动态调整日志级别, 以适应当前的需求.
    * 
    */

    return 0;
}
