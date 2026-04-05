/**
 * @file 07_interface.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 暗度陈仓: 接口与实现分离的核心
 * @version 0.1
 * @date 2026-04-05
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#ifndef LOGGER_H
#    define LOGGER_H

#    include <stdarg.h>

// 日志级别
typedef enum
{
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

// 不透明句柄：用户只能看到指针，无法访问内部结构
typedef struct Logger Logger;

// 创建日志器（传入参数暗示行为，但内部可能暗度陈仓）
Logger *logger_create(const char *target); // target 可以是文件名、udp地址等

// 销毁日志器
void logger_destroy(Logger *logger);

// 写入日志（可变参数，类似 printf）
void log_write(Logger *logger, LogLevel level, const char *fmt, ...);

// 便捷宏：隐藏级别参数
#    define LOG_INFO(logger, fmt, ...) log_write(logger, LOG_INFO, fmt, ##__VA_ARGS__)

#endif