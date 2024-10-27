/**
 * @file 21_mapped_context.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/mdc.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

// ------------------------------------
int main(int argc, const char **argv)
{
    /* 22. Mapped Diagnostic Context 附加特定的上下文信息（如用户 ID、会话 ID 等）
    * Mapped Diagnostic Context (MDC) 是 spdlog 提供的一种机制,
    * 用于在日志记录过程中自动附加特定的上下文信息(如用户 ID、会话 ID 等)到日志消息中;
    * MDC 是基于线程局部存储的, 即每个线程维护自己的 MDC 数据,
    * 因此在多线程环境中可以为每个线程附加不同的上下文信息.
    ? 不过, 由于 MDC 依赖于线程局部存储, 因此它在异步模式下不可用.
    *
    * Mapped Diagnostic Context (MDC) is a map that stores key-value pairs
    *  (string values) in thread local storage.
    * Each thread maintains its own MDC, which loggers use to append 
    * diagnostic information to log outputs.
    * Note: it is not supported in asynchronous mode due to 
    * its reliance on thread-local storage.
    */

    // 向 MDC 添加键值对
    spdlog::mdc::put("key1", "value1");
    spdlog::mdc::put("key2", "value2");

    // 设置日志格式，包括 MDC 数据（使用 %& 格式符）
    spdlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [%&] %v");

    // 创建一个日志记录器
    auto logger = spdlog::stdout_color_mt("mdc_logger");

    // 记录一些日志消息
    logger->info("This is an info message with MDC context.");
    logger->warn("This is a warning message with MDC context.");

    // 移除 MDC 中的一个键
    spdlog::mdc::remove("key1");

    // 记录另一条日志消息
    logger->error("This is an error message with modified MDC context.");

    // 清除 MDC
    spdlog::mdc::clear();

    /* 代码解释:
    * 1. 使用 MDC 添加上下文信息:
    * ----spdlog::mdc::put("key1", "value1"); 
    *   向当前线程的 MDC 中添加一个键值对 "key1" : "value1";
    * ----spdlog::mdc::put("key2", "value2"); 再次添加另一个键值对 "key2" : "value2";
    * 2. 设置日志格式:
    * spdlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [%&] %v"); 设置日志消息的格式;
    * --- %& 是用于输出 MDC 数据的格式符. 日志消息中会附加当前线程 MDC 中所有的键值对;
    * 3. 记录日志消息:
    * 使用 logger->info() 和 logger->warn() 记录日志消息.
    * ---由于 MDC 中存在键值对, 日志消息中会包含这些信息;
    * 4. 移除和修改 MDC 信息:
    * ----spdlog::mdc::remove("key1"); 从 MDC 中移除 "key1" 键及其对应的值;
    *   记录新的日志消息时, MDC 中只包含 "key2" 键值对;
    * -----清除 MDC:
    *   spdlog::mdc::clear(); 清除当前线程 MDC 中的所有键值对，恢复 MDC 到初始状态;
    * 
    * 注意事项:
    * --线程局部存储: MDC 使用线程局部存储, 因此每个线程拥有自己独立的 MDC 数据,
    *   它不能在异步模式下使用, 因为异步模式可能涉及不同的线程处理日志消息.
    * --上下文信息管理: MDC 提供了一种方便的方式来在多线程应用程序中为每个线程附加和管理上下文信息,
    *   特别适合需要详细上下文信息的日志记录场景, 如处理多个用户请求的服务器应用程序.
    * 
    * 总结: Mapped Diagnostic Context (MDC) 是 spdlog 提供的一种机制,
    *  用于在日志记录中附加线程级别的上下文信息.
    *  这对于调试和监控复杂的多线程应用程序非常有用,
    *  因为它允许在日志中包含丰富的上下文信息, 帮助更好地理解和分析日志记录的背景.
    *  通过 MDC, 可以轻松地管理和输出每个线程的特定上下文数据.
    * 
    */

    return 0;
}
