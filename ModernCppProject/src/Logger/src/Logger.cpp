#include "Logger/Logger.hpp"

#include <filesystem>
#include <format>
#include <sstream>

namespace IthacaLogger {

class LoggerImpl final : public Logger
{
public:
    void InitLogger(const std::string &log_filename, const std::string &log_name, bool isConsoleSink = false,
                    const LogLevel &level = LogLevel::TRACE, const std::string &log_folder = "logs/") noexcept override;
    void SetLogLevel(const LogLevel &log_level) noexcept override;

    void Trace(const std::string_view &message) noexcept override;
    void Debug(const std::string_view &message) noexcept override;
    void Info(const std::string_view &message) noexcept override;
    void Warn(const std::string_view &message) noexcept override;
    void Error(const std::string_view &message) noexcept override;
    void Critical(const std::string_view &message) noexcept override;

    void Trace(const char *file, int line, const char *function, const std::string_view &message) noexcept override;
    void Debug(const char *file, int line, const char *function, const std::string_view &message) noexcept override;
    void Info(const char *file, int line, const char *function, const std::string_view &message) noexcept override;
    void Warn(const char *file, int line, const char *function, const std::string_view &message) noexcept override;
    void Error(const char *file, int line, const char *function, const std::string_view &message) noexcept override;
    void Critical(const char *file, int line, const char *function, const std::string_view &message) noexcept override;

private:
    // TODO: maybe we need compression-log function
    // std::shared_ptr<spdlog::async_logger>         m_pLogger;
    std::string m_logFolder;
    std::string m_logFilename;
    std::string m_logName;
    LogLevel    m_logLevel;
    std::string m_formatter_pattern;
    // std::shared_ptr<spdlog::details::thread_pool> m_tp;

public:
    // compiler specify member-function(six default function)
    explicit LoggerImpl(const std::string &log_filename = "default_log.log",
                        const std::string &log_name = "DefaultLogger", bool isConsoleSink = false)
        : m_logFolder{"logs/"}
        , m_logFilename{log_filename}
        , m_logName{log_name}
        , m_logLevel{LogLevel::TRACE} // , m_pLogger{nullptr}
        , m_formatter_pattern{"%Y-%m-%d %H:%M:%S.%e <process %P>-<thread %t> [%n] [%l] [%@]-[%!]-[%o] %v"}
    {
        InitLogger(m_logFilename, m_logName, isConsoleSink, m_logLevel, m_logFolder);
    }

    explicit LoggerImpl(const std::string &log_filename = "default_file.log", bool isConsoleSink = false)
        : m_logFolder{"logs/"}
        , m_logFilename{log_filename}
        , m_logName{}
        , m_logLevel{LogLevel::TRACE} // , m_pLogger{nullptr}
        , m_formatter_pattern{"%Y-%m-%d %H:%M:%S.%e <process %P>-<thread %t> [%n] [%l] [%@]-[%!]-[%o] %v"}
    {
        m_logName = std::filesystem::path(m_logFilename).stem().string();

        InitLogger(m_logFilename, m_logName, isConsoleSink, m_logLevel, m_logFolder);
    }

    ~LoggerImpl() {};

    LoggerImpl(const LoggerImpl &other)            = delete;
    LoggerImpl &operator=(const LoggerImpl &other) = delete;
    LoggerImpl(LoggerImpl &&other)                 = delete; // since C++11
    LoggerImpl &operator=(LoggerImpl &&other)      = delete; // since C++11
    //LoggerImpl *operator&() {} // default compiler
    //const LoggerImpl *operator&() const {} // default compiler
};

void LoggerImpl::InitLogger(const std::string &log_filename, const std::string &log_name, bool isConsoleSink,
                            const LogLevel &level, const std::string &log_folder) noexcept
{
    m_logFolder   = log_folder;
    m_logFilename = log_filename;
    m_logName     = log_name;
    m_logLevel    = level;

    if (!std::filesystem::exists(m_logFolder))
        std::filesystem::create_directories(m_logFolder);

    auto default_logfile = m_logFolder + "/" + m_logFilename;
}

void LoggerImpl::SetLogLevel(const LogLevel &log_level) noexcept
{
    m_logLevel = log_level;
}

void LoggerImpl::Trace(const std::string_view &message) noexcept {}

void LoggerImpl::Debug(const std::string_view &message) noexcept {}

void LoggerImpl::Info(const std::string_view &message) noexcept {}

void LoggerImpl::Warn(const std::string_view &message) noexcept {}

void LoggerImpl::Error(const std::string_view &message) noexcept {}

void LoggerImpl::Critical(const std::string_view &message) noexcept {}

void LoggerImpl::Trace(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

void LoggerImpl::Debug(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

void LoggerImpl::Info(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

void LoggerImpl::Warn(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

void LoggerImpl::Error(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

void LoggerImpl::Critical(const char *file, int line, const char *function, const std::string_view &message) noexcept {}

Logger &Logger::Instance()
{
    static LoggerImpl instance = LoggerImpl("default_file.log", false);
    return instance;
}

std::shared_ptr<Logger> CreateLogger(const std::string &log_filename, const std::string &log_name,
                                     bool isConsoleSink) noexcept
{
    return std::shared_ptr<LoggerImpl>{new LoggerImpl(log_filename, log_name, isConsoleSink)};
}

} // namespace IthacaLogger