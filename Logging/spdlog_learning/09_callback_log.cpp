/**
 * @file 09_callback_log.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/callback_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <iostream>
#include <memory>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 10. User-defined callbacks about log events 用户定义回调
    * spdlog 还允许用户定义回调函数, 用于在日志事件发生时执行自定义操作.
    * 这种功能在需要对特定的日志事件采取即时措施时非常有用, 
    * 例如发送警报、通知、或者执行其他逻辑.
    * 
    * 特性介绍:
    * 1. 用户定义回调: 可以为日志记录器设置一个回调函数, 
    *    当满足特定条件的日志消息被记录时, 该回调函数会被调用;
    *    这允许开发者在日志记录时执行额外的操作, 例如发送通知或执行自定义的业务逻辑.
    * 2. 灵活性: 回调函数可以根据日志级别、内容或其他条件来触发不同的操作, 这使得日志处理更加灵活和强大.
    * 
    */
    // 创建一个回调 sink
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [](const spdlog::details::log_msg &msg)
        {
            // 自定义回调逻辑，例如发送通知或邮件
            std::string log_message(msg.payload.begin(), msg.payload.end());
            // 在这里，你可以执行任意操作，例如发送邮件、记录到数据库、调用外部API等。
            std::cout << "Callback triggered: " << log_message << std::endl;
        });

    // 设置回调 sink 只处理 error 级别及以上的日志
    callback_sink->set_level(spdlog::level::err);

    // 创建一个控制台 sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    // 创建一个包含多个 sinks 的 logger
    spdlog::logger logger("custom_callback_logger", {console_sink, callback_sink});

    // 记录不同级别的日志
    logger.info("some info log");   // 只会输出到控制台，不触发回调
    logger.error("critical issue"); // 会输出到控制台并触发回调

    /* 代码解释:
    * 1. 创建回调 sink:
    * ----使用 spdlog::sinks::callback_sink_mt 创建一个多线程安全的回调 sink,
    *    并传入一个 lambda 函数作为回调, 这段代码中的回调函数简单地将日志消息输出到控制台,
    *    但可以在此处添加更复杂的逻辑, 如发送邮件或发送到GUI界面或执行其他操作.
    * ----msg 是 spdlog::details::log_msg 类型的结构, 包含了日志消息及其元数据,
    *   可以使用它来获取消息内容或其他相关信息.
    * 2. 设置日志级别: 使用 callback_sink->set_level(spdlog::level::err); 
    *   设置回调 sink 只处理 error 级别及以上的日志,
    *   这意味着只有在记录 error 或更严重的日志时, 回调函数才会被触发.
    * 3. 创建控制台 sink: stdout_color_sink_mt 用于将日志输出到控制台, 并且支持彩色输出.
    * 4. 创建带有多个 sinks 的 logger:
    * ----spdlog::logger logger("custom_callback_logger", {console_sink, callback_sink}); 
    *    创建了一个包含两个 sinks 的日志记录器. 
    *    这个记录器会将日志输出到控制台, 同时在满足条件时触发回调.
    * 5. 记录日志:
    * ----info 级别的日志只会输出到控制台;
    * ----error 级别的日志既会输出到控制台, 又会触发回调;
    * 
    * 总结: 通过使用 spdlog 的回调功能, 可以在记录日志的同时, 执行自定义的操作,
    * 如发送通知、执行额外的业务逻辑等.
    * 这使得 spdlog 不仅仅是一个简单的日志库, 还可以用作事件驱动的工具,
    * 在特定的日志事件发生时, 自动触发相应的响应措施.
    * 这种能力对于实时监控、报警系统或任何需要对特定日志事件采取行动的应用程序非常有用.
    * 
    */

    return 0;
}
