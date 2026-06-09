/**
 * @file Logger.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Logger utility class for application logging
 * @version 0.1
 * @date 2026-04-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Core/SymbolExport.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace IthacaLogger {

enum class LogLevel
{
    TRACE = 0X0000,
    DEBUG = 0X0001,
    INFO,
    WARN,
    ERR,
    CRITICAL,
};

class LIB_API Logger
{
public:
    /**
     * @brief 初始化日志, 设置日志储存的路径文件夹, 日志文件名, 日志名称, 以及是否需要 console-sink
     *
     * @param [IN]log_name, register logger according to 'log_name', and get logger instance with pointer.
     * @param [IN]log_filename, the local file sink log-filename, default postfix=='.log'.
     * @param [IN]log_folder, the folder path of log file store in the disk, the default value=="logs/".
     * @param [IN]level, the log-level, the default value=="TRACE".
     * @param [IN]isConsoleSink, the default value==false.
     * @return void
     * @note
     */
    virtual void InitLogger(const std::string &log_filename, const std::string &log_name, bool isConsoleSink = false,
                            const LogLevel &level = LogLevel::TRACE, const std::string &log_folder = "logs/") noexcept
        = 0;

    /**
     * @brief 设置日志等级level
     *
     * @param [IN]log_level, set logger level.
     * @return void
     * @note
     */
    virtual void SetLogLevel(const LogLevel &log_level) noexcept = 0;

    /**
     * @brief 根据日志等级, 输出日志内容
     *
     * @param [IN]message, the log message context.
     * @return void
     * @note
     */
    virtual void Trace(const std::string_view &message) noexcept    = 0;
    virtual void Debug(const std::string_view &message) noexcept    = 0;
    virtual void Info(const std::string_view &message) noexcept     = 0;
    virtual void Warn(const std::string_view &message) noexcept     = 0;
    virtual void Error(const std::string_view &message) noexcept    = 0;
    virtual void Critical(const std::string_view &message) noexcept = 0;

    /**
     * @brief 根据日志等级, 输出日志内容
     *
     * @param [IN]message, the log message context.
     * @param [IN]file, 外部使用 宏 __FILE__ 记录源代码文件路径
     * @param [IN]line, 外部使用 宏 __LINE__ 记录源代码-行号路径
     * @param [IN]function, 外部使用 宏 __FUNCTION__ 记录函数名
     * @return void
     * @note
     */
    virtual void Trace(const char *file, int line, const char *function, const std::string_view &message) noexcept = 0;
    virtual void Debug(const char *file, int line, const char *function, const std::string_view &message) noexcept = 0;
    virtual void Info(const char *file, int line, const char *function, const std::string_view &message) noexcept  = 0;
    virtual void Warn(const char *file, int line, const char *function, const std::string_view &message) noexcept  = 0;
    virtual void Error(const char *file, int line, const char *function, const std::string_view &message) noexcept = 0;
    virtual void Critical(const char *file, int line, const char *function, const std::string_view &message) noexcept
        = 0;

    /**
     * @brief 静态实例类
     * @note
     */
    static Logger &Instance();

public:
    virtual ~Logger() {}
};

/**
 * @brief Create 'LoggerInterface' Instance
 * The function CreateLogger create instance logger.
 * @param [IN]log_filename, the local file sink log-filename, default postfix=='.log'.
 * @param [IN]log_name, the global unique-key for name.
 * @return std::shared_ptr<CameraInterface>
 */
[[nodiscard]] LIB_API std::shared_ptr<Logger> CALLING_CONVENTIONS CreateLogger(const std::string &log_filename,
                                                                               const std::string &log_name,
                                                                               bool isConsoleSink = false) noexcept;

} // namespace IthacaLogger