#ifndef __LOGGING_HPP__
#define __LOGGING_HPP__

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

// 首先需要定义日志的几种级别，并将其与日志消息、时间戳等信息封装在一起
// 日志级别
enum class LogLevel : char
{
    INFO      = 0,
    WARNING   = 1,
    ERROR     = 2,
    EXCEPTION = 3,
    NUM_LEVEL
};

// 日志项
struct LogEntry
{
    std::string message;
    LogLevel    level;
    std::string timestamp;
};

// 获取当前时间的时间戳
std::string GetCurrentTimestamp();
// 获取当前日期，用于日志文件命名
std::string GetCurrentDate();

// 构建一个 MyLogger 类，负责处理日志的异步写入,
// 为了避免阻塞主线程, 将日志写入操作放在单独的线程中处理,
// 并使用 std::queue 存储待写入的日志
class CustomLogger
{
public:
    CustomLogger(const std::string &output_dir)
        : output_dir_(output_dir)
        , stop_flag_(false)
    {
        StartLoggingThread();
    }

    ~CustomLogger()
    {
        StopLoggingThread();
    }

    // 记录日志
    void Log(const std::string &message, LogLevel level)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        log_queue_.emplace(LogEntry{message, level, GetCurrentTimestamp()});
        condition_.notify_one(); // 通知日志线程有新日志
    }

private:
    std::string             output_dir_;
    std::queue<LogEntry>    log_queue_;
    std::mutex              queue_mutex_;
    std::condition_variable condition_;
    bool                    stop_flag_;
    std::thread             logging_thread_;

private:
    // 停止日志线程
    void StopLoggingThread();
    // 启动日志线程
    void StartLoggingThread();

    // 写入日志到文件
    inline void WriteLogToFile(const LogEntry &entry);

    // 日志等级枚举量转化为字符串
    inline std::string LogLevelToString(const LogLevel &level);
};

class CustomLoggerBenchmark
{
public:
    CustomLoggerBenchmark(CustomLogger &logger)
        : logger_(logger)
    {
    }

    // 单次日志写入测试
    void SingleLogTest();
    // 批量日志写入测试
    void BulkLogTest(int num_logs);

private:
    CustomLogger &logger_;
};

#endif /* __LOGGING_HPP__ */
