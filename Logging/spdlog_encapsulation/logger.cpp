#include "logger.hpp"

void Logger::Init(const LogConfig &conf)
{
    //自定义的sink
    loggerPtr = spdlog::rotating_logger_mt("base_logger", conf.path.c_str(), conf.size, conf.count);
    //设置格式
    //参见文档 https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    //[%Y-%m-%d %H:%M:%S.%e] 时间
    //[%l] 日志级别
    //[%t] 线程
    //[%s] 文件
    //[%#] 行号
    //[%!] 函数
    //[%v] 实际文本
    loggerPtr->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] [%s %!:%#] %v");

    // 设置日志级别
    loggerPtr->set_level(spdlog::level::from_str(conf.level));
    // 设置刷新日志的日志级别，当出现level或更高级别日志时，立刻刷新日志到  disk
    loggerPtr->flush_on(spdlog::level::from_str(conf.level));
}

/*
 * trace 0
 * debug 1
 * info 2
 * warn 3
 * error 4
 * critical 5
 * off 6 (not use)
 */
std::string Logger::GetLogLevel()
{
    auto level = loggerPtr->level();
    return spdlog::level::to_string_view(level).data();
}

void Logger::SetLogLevel(const std::string &log_level)
{
    auto level = spdlog::level::from_str(log_level);
    if (level == spdlog::level::off)
    {
        WARNLOG("Given invalid log level {}", log_level);
    }
    else
    {
        loggerPtr->set_level(level);
        loggerPtr->flush_on(level);
    }
}
