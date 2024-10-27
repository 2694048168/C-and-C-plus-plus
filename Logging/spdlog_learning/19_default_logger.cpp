/**
 * @file 19_default_logger.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 20. Replace the Default Logger 替换默认的日志记录器
    * 在 spdlog 中, 可以通过替换默认的日志记录器来更改日志记录的行为.
    * 默认日志记录器通常用于那些不指定特定日志记录器的日志消息.
    * 通过替换默认日志记录器, 可以将日志消息定向到不同的输出目标或文件,并自定义日志记录的格式和级别.
    */

    // 创建一个新的日志记录器，并将其作为默认日志记录器
    auto new_logger = spdlog::basic_logger_mt("new_default_logger", "logs/new_default_log.txt", true);

    // 设置新的日志记录器为默认日志记录器
    spdlog::set_default_logger(new_logger);

    // 使用默认日志记录器记录消息
    spdlog::info("new logger log message");
    spdlog::debug("new logger log message");
    spdlog::warn("new logger log message");
    spdlog::error("new logger log message");
    spdlog::critical("new logger log message");

    /* 代码解释:
    * 1. 创建新的日志记录器:
    * ----spdlog::basic_logger_mt 创建了一个新的日志记录器,
    *   该记录器会将日志消息写入 logs/new-default-log.txt 文件中.
    *   "new_default_logger" 是日志记录器的名称.
    *   true 参数表示以追加模式打开日志文件（如果文件已存在，则在文件末尾追加新日志）.
    * 2. 设置新的默认日志记录器:
    * ----spdlog::set_default_logger(new_logger); 
    *   将刚刚创建的 new_logger 设置为默认的日志记录器;
    *   这意味着任何未显式指定记录器的日志消息都将使用这个新的记录器;
    * 3. 记录日志消息:
    * ----这条消息将使用新的默认日志记录器, 并被记录到 logs/new-default-log.txt 文件中.
    * 
    */

    return 0;
}
