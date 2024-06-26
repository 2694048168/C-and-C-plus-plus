/**
 * @file logger.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 对 spdlog 日志库进行封装, 便于集成和使用 
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

// 日志的配置项
struct LogConfig
{
    std::string level;
    std::string path;
    int64_t     size;
    int         count;
};

// 日志的单例模式
class Logger
{
public:
    static Logger *getInstance()
    {
        static Logger instance;
        return &instance;
    }

    //c++14返回值可设置为auto
    std::shared_ptr<spdlog::logger> getLogger()
    {
        return loggerPtr;
    }

    void Init(const LogConfig &conf);

    std::string GetLogLevel();

    void SetLogLevel(const std::string &level);

private:
    Logger() = default;
    std::shared_ptr<spdlog::logger> loggerPtr;
};

// 日志相关操作的宏封装
#define INITLOG(conf)               Logger::getInstance()->Init(conf)
#define GETLOGLEVEL()               Logger::getInstance()->GetLogLevel()
#define SETLOGLEVEL(level)          Logger::getInstance()->SetLogLevel(level)
#define BASELOG(logger, level, ...) (logger)->log(spdlog::source_loc{__FILE__, __LINE__, __func__}, level, __VA_ARGS__)
#define TRACELOG(...)               BASELOG(Logger::getInstance()->getLogger(), spdlog::level::trace, __VA_ARGS__)
#define DEBUGLOG(...)               BASELOG(Logger::getInstance()->getLogger(), spdlog::level::debug, __VA_ARGS__)
#define INFOLOG(...)                BASELOG(Logger::getInstance()->getLogger(), spdlog::level::info, __VA_ARGS__)
#define WARNLOG(...)                BASELOG(Logger::getInstance()->getLogger(), spdlog::level::warn, __VA_ARGS__)
#define ERRORLOG(...)               BASELOG(Logger::getInstance()->getLogger(), spdlog::level::err, __VA_ARGS__)
#define CRITICALLOG(...)            BASELOG(Logger::getInstance()->getLogger(), spdlog::level::critical, __VA_ARGS__)
