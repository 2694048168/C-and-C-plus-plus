/**
 * @file main_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief spdlog 日志库封装后使用例子
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "logger.hpp"

// --------------------------------------
int main(int argc, const char **argv)
{
    // 定义日志配置项
    LogConfig conf = {
        .level = "trace",
        .path  = "logger_test.log",
        .size  = 5 * 1024 * 1024,
        .count = 10,
    };
    INITLOG(conf);
    // 日志初始级别为trace
    TRACELOG("current log level is {}", GETLOGLEVEL());
    TRACELOG("this is trace log");
    DEBUGLOG("this is debug log");
    INFOLOG("this is info log");
    WARNLOG("this is warning log");
    ERRORLOG("this is a error log");
    CRITICALLOG("this is critical log");

    // 改为warning级别后，trace、debug、info级别日志不会输出了
    SETLOGLEVEL("warn");
    WARNLOG("after set log level to warning");
    TRACELOG("this is trace log");
    DEBUGLOG("this is debug log");
    INFOLOG("this is info log");
    WARNLOG("this is warning log");
    ERRORLOG("this is a error log");
    CRITICALLOG("this is critical log");

    return 0;
}
