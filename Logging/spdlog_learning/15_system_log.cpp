/**
 * @file 15_system_log.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/syslog_sink.h"
#include "spdlog/spdlog.h"

#include <string>

int main(int argc, const char **argv)
{
    /* 16 syslog 系统日志
    * 在 spdlog 中, 可以使用 syslog_sink 将日志消息发送到系统日志(syslog).
    * syslog 是 Unix 系统中一个重要的日志记录工具, 用于记录系统事件和应用程序日志.
    * 通过将日志消息发送到 syslog, 可以将其集成到系统的日志管理基础设施中, 方便统一管理和监控.
    */

    // 定义 syslog 的标识符
    std::string ident = "spdlog_example";

    // 创建一个 syslog 日志记录器
    auto syslog_logger = spdlog::syslog_logger_mt("syslog", ident, LOG_PID);

    // 记录一个警告信息到 syslog
    syslog_logger->warn("This is a warning that will end up in syslog.");

    /* 代码解释
    * 1. 包含必要的头文件:
    * ----spdlog/spdlog.h 是 spdlog 的主头文件, 包含了日志记录功能的核心内容;
    * ----spdlog/sinks/syslog_sink.h 包含了 syslog_sink, 用于将日志消息发送到 syslog;
    * 2. 定义 syslog 标识符:
    * ----std::string ident = "spdlog-example"; 定义了发送到 syslog 的标识符（ident）;
    *  这个标识符将在 syslog 中标识出哪个程序或模块生成了日志消息.
    * 3. 创建 syslog 日志记录器:
    * ----spdlog::syslog_logger_mt("syslog", ident, LOG_PID); 创建了一个 syslog 日志记录器.
    * ----"syslog" 是日志记录器的名称; ident 是标识符; 
    * ----LOG_PID 是 syslog 的选项, 用于在日志消息中包含生成日志的进程 ID.
    * 4. 记录日志到 syslog:
    * 
    * 输出结果, 在运行上述代码后, 如果在 Unix 系统上配置正确,
    * 可以在 syslog 文件中（通常是 /var/log/syslog 或 /var/log/messages）看到类似以下的日志消息:
    * Aug 11 12:34:56 hostname spdlog-example[12345]: This is a warning that will end up in syslog.
    * ----hostname 是系统的主机名;
    * ----spdlog-example 是我们设置的 ident 标识符;
    * ----[12345] 是生成此日志的进程 ID;
    * ----This is a warning that will end up in syslog. 是日志消息的内容;
    * 通过使用 spdlog 的 syslog_sink, 可以将应用程序的日志消息发送到 Unix 系统的 syslog,
    * 从而利用系统级的日志管理和监控工具. 这样可以将应用程序日志与系统日志统一管理,
    * 方便运维人员进行监控和故障排查. 
    * syslog 是 Unix 系统中广泛使用的日志记录机制, 
    * 将日志消息集成到 syslog 中能够提高日志管理的集中性和效率.
    * 
    */

    return 0;
}
