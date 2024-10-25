/**
 * @file 08_multi_sinks.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

// ------------------------------------
int main(int argc, const char **argv)
{
    /* 9. Logger with multi sinks - each with a different format and log level 设置日志级别
    * spdlog 支持创建具有多个 "sinks"（日志输出目标）的日志记录器,
    * 并且每个 sink 可以有不同的格式和日志级别.
    * 这种能力使得开发者可以灵活地将日志输出到不同的目标(如文件、控制台等),
    * 并为每个目标设置不同的日志级别和格式, 以满足不同的日志记录需求. 
    * 
    * 实现步骤:
    * 1. 创建多个 sinks: 每个 sink 可以指定不同的输出目标(如文件、控制台)以及不同的格式和日志级别;
    * 2. 将这些 sinks 组合成一个 logger: 这个 logger 将日志同时输出到多个目标;
    * 3. 使用 logger: 当日志消息被记录时, 它会根据每个 sink 的设置输出到相应的目标;
    * 
    */
    // 创建控制台 sink，并设置日志级别为 info
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");

    // 创建文件 sink，并设置日志级别为 debug
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multi_sink_log.txt", true);
    file_sink->set_level(spdlog::level::debug);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [thread %t] %v");

    // 创建一个带有多个 sinks 的 logger
    spdlog::logger logger("multi_sink", {console_sink, file_sink});

    // 设置 logger 的日志级别为 debug（这样所有级别的日志都能被输出）
    logger.set_level(spdlog::level::debug);

    // 使用 logger 记录不同级别的日志
    logger.debug("This is a debug message");
    logger.info("This is an info message");
    logger.warn("This is a warning message");
    logger.error("This is an error message");

    // 刷新日志（将缓冲区的内容写入目标）
    spdlog::shutdown();

    /* 代码解释:
    * 1. 创建控制台 sink:
    * ----使用 stdout_color_sink_mt 创建了一个多线程安全的彩色控制台输出 sink;
    * ----设置了日志级别为 info, 表示只有 info 级别及更高级别的日志会输出到控制台;
    * ----使用 set_pattern 方法设置了日志的输出格式，例如时间、日志级别、消息内容等;
    * 2. 创建文件 sink:
    * ----使用 basic_file_sink_mt 创建了一个文件输出 sink, 日志会被记录到 
    *     logs/multi_sink_log.txt 文件中;
    * ----设置了日志级别为 debug, 这样即使是调试信息也会被写入到文件中;
    * ----同样使用 set_pattern 设置了文件日志的格式，包含日期、时间、线程 ID 等信息;
    * 3. 创建带有多个 sinks 的 logger:
    * ----使用 spdlog::logger 创建了一个 logger, 并将之前创建的控制台和文件 sinks 传递给它;
    * ----设置 logger 的日志级别为 debug, 确保所有级别的日志消息都会被发送到各个 sink;
    * 4. 记录日志:
    * ----使用 logger 记录了不同级别的日志消息;
    * ----由于控制台 sink 的日志级别为 info, 因此 debug 级别的消息不会显示在控制台上,但会写入文件;
    * 5. 日志刷新与关闭:
    * -----在程序结束前, 使用 spdlog::shutdown() 确保所有日志数据被正确刷新到目标位置;
    * 输出示例:
    * ----控制台输出, 由于控制台的日志级别设置为 info, 因此 debug 消息不会显示;
    * ----文件输出, 文件的日志级别设置为 debug, 所以所有日志消息都被记录到文件中;
    * 总结: 通过这种方式, 可以在一个日志记录器中同时使用多个 sinks,
    * 并为每个 sink 定制不同的日志级别和格式, 这样可以满足不同的日志记录需求;
    * 比如将详细的调试信息保存到文件中, 同时在控制台上只显示更为重要的日志消息;
    * 这种多 sink 的配置在复杂应用程序中非常实用, 尤其是在需要将日志输出到多个目标时,
    * 如同时记录到控制台、文件、远程服务器等.
    * 
    */

    return 0;
}
