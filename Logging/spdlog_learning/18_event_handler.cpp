/**
 * @file 18_event_handler.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

// -------------------------------------
int main(int argc, const char **argv)
{
    /* 19. Log file open/close event handlers 打开和关闭事件注册回调函数
    * 在 spdlog 中, 可以为日志文件的打开和关闭事件注册回调函数.
    * 这些回调函数允许在日志文件打开之前、打开之后、关闭之前以及关闭之后执行特定的操作.
    * 例如可以在日志文件打开后添加一些初始信息, 或者在日志文件关闭前执行清理工作.
    */

    // 初始化 spdlog
    spdlog::set_level(spdlog::level::info);                         // 设置日志级别
    spdlog::set_default_logger(spdlog::stdout_color_mt("console")); // 使用彩色控制台输出

    // 创建文件事件处理程序
    spdlog::file_event_handlers handlers;

    // 在日志文件打开之前触发的回调
    handlers.before_open = [](spdlog::filename_t filename)
    {
        spdlog::info("Before opening {}", filename);
    };

    // 在日志文件打开之后触发的回调
    handlers.after_open = [](spdlog::filename_t filename, std::FILE *fstream)
    {
        fputs("After opening\n", fstream); // 在日志文件中添加一行
    };

    // 在日志文件关闭之前触发的回调
    handlers.before_close = [](spdlog::filename_t filename, std::FILE *fstream)
    {
        fputs("Before closing\n", fstream); // 在日志文件中添加一行
    };

    // 在日志文件关闭之后触发的回调
    handlers.after_close = [](spdlog::filename_t filename)
    {
        spdlog::info("After closing {}", filename);
    };

    // 创建一个带有事件处理程序的基本文件日志记录器
    auto my_logger = spdlog::basic_logger_st("some_logger", "logs/events_sample.txt", true, handlers);

    // 记录一些日志消息
    my_logger->info("This is an info message.");
    my_logger->warn("This is a warning message.");

    // 显式关闭日志记录器，以触发关闭事件
    my_logger->flush();
    spdlog::drop("some_logger"); // 删除日志记录器并关闭日志文件

    /* 代码解释:
    * 1. 创建文件事件处理程序:
    * ----spdlog::file_event_handlers handlers; 
    *   定义了一个文件事件处理程序对象, 用于处理文件的打开和关闭事件;
    * ----handlers.before_open 在日志文件打开之前执行, 记录文件打开前的操作;
    * ----handlers.after_open 在日志文件打开之后执行, 在文件中添加初始内容（如 "After opening\n"）;
    * ----handlers.before_close 在日志文件关闭之前执行,添加内容（如 "Before closing\n"）;
    * ----handlers.after_close 在日志文件关闭之后执行, 记录文件关闭后的操作;
    * 2. 创建带有事件处理程序的文件日志记录器:
    * ----spdlog::basic_logger_st 创建了一个基本的文件日志记录器,并将事件处理程序传递给它;
    *   "logs/events-sample.txt" 是日志文件的路径, true 表示以追加模式打开文件,
    * 3. 记录日志消息并关闭日志文件:
    * ----通过 my_logger->info() 和 my_logger->warn() 记录日志消息;
    * ----使用 my_logger->flush() 和 spdlog::drop("some_logger") 来
    *   显式关闭日志记录器, 这会触发 before_close 和 after_close 回调;
    *
    * 总结: 通过使用 spdlog::file_event_handlers,
    * 可以灵活地控制和监控日志文件的打开和关闭事件.
    * 这在需要在日志文件生命周期内执行特定操作（如记录文件打开/关闭事件、添加文件头或尾信息等）时非常有用,
    * 这种机制增强了日志管理的灵活性和可扩展性, 使得 spdlog 能够适应更复杂的日志记录需求.
    * 
    */

    return 0;
}
