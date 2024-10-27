/**
 * @file Logger.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 * 封装Log头文件
 * *一般的项目对日志要求都不高,主要是要求
 * 日志线程安全、异步写入文件、每天生成新日志、支持日志回调显示，spdlog稍微配置一下即可.
 * 把spdlog相关的配置全放到Logger.h文件中, 封装成Logger头文件有两个好处:
 * 1. 可以随时替换后台日志实现;
 * 2. 对外只用暴露一个头文件;
 * 
 */

#pragma once

#include "spdlog/async.h"
#include "spdlog/sinks/callback_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

void init_spdlog()
{
    //异步日志，具有8k个项目和1个后台线程的队列
    spdlog::init_thread_pool(8192, 1);

    //标准控制台输出
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    stdout_sink->set_level(spdlog::level::debug);

    //日志文件输出，0点0分创建新日志
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/log.txt", 0, 0);
    file_sink->set_level(spdlog::level::info);

    //日志回调
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [](const spdlog::details::log_msg &msg)
        {
            //日志记录器名称
            std::string name(msg.logger_name.data(), 0, msg.logger_name.size());
            //日志消息
            std::string str(msg.payload.data(), 0, msg.payload.size());
            //日志时间
            std::time_t now_c = std::chrono::system_clock::to_time_t(msg.time);

            //回调的处理逻辑自己根据项目情况定义，比如实时显示到UI、保存到数据库等等

            //.... 回调处理逻辑的示例
            //std::tm localTime;
            //localtime_s(&localTime, &now_c);
            //char timeStr[50];
            //std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &localTime);
            //// 获取毫秒数
            //auto duration = msg.time.time_since_epoch();
            //auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
            //std::cout << timeStr << "." << std::setfill('0') << std::setw(3) << milliseconds << " " ;
            //std::cout << to_string_view(msg.level).data() << " " << str << std::endl << std::endl << std::flush;
        });
    callback_sink->set_level(spdlog::level::info);

    std::vector<spdlog::sink_ptr> sinks{stdout_sink, file_sink, callback_sink};
    auto log = std::make_shared<spdlog::async_logger>("logger", sinks.begin(), sinks.end(), spdlog::thread_pool(),
                                                      spdlog::async_overflow_policy::block);

    //设置日志记录级别，您需要用 %^ 和 %$  括上想要彩色的部分
    log->set_level(spdlog::level::trace);
    //设置格式
    //参考 https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    //[%Y-%m-%d %H:%M:%S.%e] 时间
    //[%l] 日志级别
    //[%t] 线程
    //[%s] 文件
    //[%#] 行号
    //[%!] 函数
    //[%v] 实际文本
    log->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l]%$ [%t] [%s %!:%#] %v");

    //设置当出发 err 或更严重的错误时立刻刷新日志到  disk
    log->flush_on(spdlog::level::err);
    
    //3秒刷新一次队列
    spdlog::flush_every(std::chrono::seconds(3));
    spdlog::set_default_logger(log);
}

//单个日志记录器
std::shared_ptr<spdlog::logger> get_async_file_logger(std::string name)
{
    auto log = spdlog::get(name);
    if (!log)
    {
        //指针为空，则创建日志记录器，
        log = spdlog::daily_logger_mt<spdlog::async_factory>(name, "logs/" + name + "/log.txt");
        log->set_level(spdlog::level::trace);
        log->flush_on(spdlog::level::err);
        log->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l]%$ [%t] [%s %!:%#] %v");
        //记录器是自动注册的，不需要手动注册  spdlog::register_logger(name);
    }
    return log;
}

#define INITLOG() init_spdlog()

#define TRACE(...)    SPDLOG_TRACE(__VA_ARGS__)
#define DEBUG(...)    SPDLOG_DEBUG(__VA_ARGS__)
#define INFO(...)     SPDLOG_INFO(__VA_ARGS__)
#define WARN(...)     SPDLOG_WARN(__VA_ARGS__)
#define ERROR(...)    SPDLOG_ERROR(__VA_ARGS__)
#define CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

//单个日志文件
#define GETLOG(LOG_NAME) get_async_file_logger(LOG_NAME)

#define LOGGER_TRACE(logger, ...)    SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#define LOGGER_DEBUG(logger, ...)    SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#define LOGGER_INFO(logger, ...)     SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#define LOGGER_WARN(logger, ...)     SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#define LOGGER_ERROR(logger, ...)    SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#define LOGGER_CRITICAL(logger, ...) SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__)

//时间统计宏
#define LOGSW() spdlog::stopwatch()
