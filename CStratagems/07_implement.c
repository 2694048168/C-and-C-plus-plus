/**
 * @file 07_implement.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 日志系统实现文件
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 07_implement.exe 07_implement.c
 * clang -o 07_implement.exe 07_implement.c
 *
 */

#include "07_interface.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

// 实际可能的输出目标
typedef enum
{
    TARGET_CONSOLE,
    TARGET_FILE,
    TARGET_NETWORK // 示例中简化，不真正实现网络
} TargetType;

// 真正的日志器结构（对外完全隐藏）
struct Logger
{
    TargetType type;
    union {
        FILE *file; // 文件指针
        // 网络相关句柄可扩展
    } u;
    char *prefix; // 可选前缀
};

// 获取当前时间字符串（内部辅助）
static void get_time_str(char *buf, size_t len)
{
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S", tm);
}

// 不同实现的具体写入函数（暗度陈仓的核心）
static void write_console(const char *msg)
{
    printf("%s\n", msg);
}

static void write_file(FILE *fp, const char *msg)
{
    fprintf(fp, "%s\n", msg);
    fflush(fp);
}

// 根据日志器类型，调用不同的底层函数
static void do_output(Logger *logger, const char *formatted_msg)
{
    switch (logger->type)
    {
    case TARGET_CONSOLE:
        write_console(formatted_msg);
        break;
    case TARGET_FILE:
        if (logger->u.file)
        {
            write_file(logger->u.file, formatted_msg);
        }
        break;
    default:
        break;
    }
}

// 创建日志器：根据 target 字符串暗度陈仓，决定实际类型
Logger *logger_create(const char *target)
{
    Logger *logger = (Logger *)malloc(sizeof(Logger));
    if (!logger)
        return NULL;

    if (target == NULL || strcmp(target, "console") == 0)
    {
        logger->type = TARGET_CONSOLE;
        logger->u.file = NULL;
    }
    else
    {
        // 假装是文件（暗度陈仓：其实也可以是网络，但这里只演示文件）
        logger->type = TARGET_FILE;
        FILE *fp = fopen(target, "a");
        if (!fp)
        {
            free(logger);
            return NULL;
        }
        logger->u.file = fp;
    }
    logger->prefix = strdup("App");
    if (!logger->prefix)
    {
        if (logger->type == TARGET_FILE)
            fclose(logger->u.file);
        free(logger);
        return NULL;
    }
    return logger;
}

void logger_destroy(Logger *logger)
{
    if (logger)
    {
        if (logger->type == TARGET_FILE && logger->u.file)
        {
            fclose(logger->u.file);
        }
        free(logger->prefix);
        free(logger);
    }
}

void log_write(Logger *logger, LogLevel level, const char *fmt, ...)
{
    if (!logger)
        return;

    const char *level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    char time_buf[64];
    get_time_str(time_buf, sizeof(time_buf));

    // 格式化用户消息
    va_list args;
    va_start(args, fmt);
    char msg_buf[1024];
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    // 组装完整日志行
    char full_line[2048];
    snprintf(full_line, sizeof(full_line), "[%s] [%s] [%s] %s", time_buf, level_str[level], logger->prefix, msg_buf);

    // 暗度陈仓：实际输出方式由创建时决定
    do_output(logger, full_line);
}

int main(int argc, char *argv[])
{
    SetConsoleOutputCP(CP_UTF8); // 设置输出代码页为 UTF-8

    // 明修栈道：用户以为创建控制台日志器
    Logger *console_log = logger_create("console");
    if (console_log)
    {
        LOG_INFO(console_log, "这是控制台输出，用户不知道内部如何实现");
        logger_destroy(console_log);
    }

    // 暗度陈仓：传入文件名，实际写入文件
    Logger *file_log = logger_create("app.log");
    if (file_log)
    {
        LOG_INFO(file_log, "这条日志写入了文件，调用者完全无感");
        logger_destroy(file_log);
    }

    return 0;
}
