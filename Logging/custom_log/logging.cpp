#include "logging.hpp"

#include <chrono>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// 获取当前时间的时间戳
std::string GetCurrentTimestamp()
{
    auto        now      = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm     tm;
    localtime_s(&tm, &now_time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// 获取当前日期，用于日志文件命名
std::string GetCurrentDate()
{
    auto        now      = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm     tm;
    localtime_s(&tm, &now_time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d");
    return ss.str();
}

// 启动日志线程
void CustomLogger::StartLoggingThread()
{
    logging_thread_ = std::thread(
        [this]()
        {
            while (!stop_flag_ || !log_queue_.empty())
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this]() { return !log_queue_.empty() || stop_flag_; });

                while (!log_queue_.empty())
                {
                    LogEntry entry = log_queue_.front();
                    log_queue_.pop();
                    lock.unlock();
                    WriteLogToFile(entry);
                    lock.lock();
                }
            }
        });
}

// 停止日志线程
void CustomLogger::StopLoggingThread()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_flag_ = true;
    }
    condition_.notify_one();
    if (logging_thread_.joinable())
    {
        logging_thread_.join();
    }
}

// 写入日志到文件
inline void CustomLogger::WriteLogToFile(const LogEntry &entry)
{
    std::string   filename = output_dir_ + "/log_" + GetCurrentDate() + ".txt";
    std::ofstream log_file(filename, std::ios_base::app);
    if (log_file.is_open())
    {
        log_file << "[" << entry.timestamp << "] " << LogLevelToString(entry.level) << ": " << entry.message
                 << std::endl;
    }
}

// 日志等级枚举量转化为字符串
inline std::string CustomLogger::LogLevelToString(const LogLevel &level)
{
    switch (level)
    {
    case LogLevel::INFO:
        return "[INFO]";
        // break;

    case LogLevel::WARNING:
        return "[WARNING]";
        // break;

    case LogLevel::ERROR:
        return "[ERROR]";
        // break;

    case LogLevel::EXCEPTION:
        return "[EXCEPTION]";
        // break;

    default:
        return "[LOG]";
        // break;
    }
}

// 单次日志写入测试
void CustomLoggerBenchmark::SingleLogTest()
{
    auto start = std::chrono::high_resolution_clock::now();
    logger_.Log("Single log test message", LogLevel::INFO);
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Single log write took " << duration << " microseconds." << std::endl;
}

// 批量日志写入测试
void CustomLoggerBenchmark::BulkLogTest(int num_logs)
{
    std::vector<std::string> messages;
    for (size_t idx{0}; idx < num_logs; ++idx)
    {
        messages.push_back("Bulk log test message #" + std::to_string(idx + 1));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &msg : messages)
    {
        logger_.Log(msg, LogLevel::INFO);
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Bulk log write of " << num_logs << " logs took " << duration << " milliseconds." << std::endl;
    std::cout << "Average log write time: " << duration * 1000.0 / num_logs << " microseconds." << std::endl;
}
