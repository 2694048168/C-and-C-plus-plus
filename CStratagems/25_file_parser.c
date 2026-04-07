/**
 * @file 25_file_parser.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 指桑骂槐: 日志和断言的技艺
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 25_file_parser.exe 25_file_parser.c
 * clang -o 25_file_parser.exe 25_file_parser.c
 *
 */

#include <Windows.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ========== 日志系统（指桑骂槐的工具） ==========
typedef enum
{
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN, // 警告：指桑，实际暗示潜在问题
    LOG_ERROR // 错误：骂槐，但程序可能继续
} LogLevel;

static LogLevel g_min_level = LOG_INFO;
static FILE *g_log_file = NULL;

void log_init(LogLevel min_level, const char *filename)
{
    g_min_level = min_level;
    if (filename)
    {
        g_log_file = fopen(filename, "a");
    }
    else
    {
        g_log_file = stderr;
    }
}

void log_close(void)
{
    if (g_log_file && g_log_file != stderr)
    {
        fclose(g_log_file);
    }
}

void log_message(LogLevel level, const char *file, int line, const char *fmt, ...)
{
    if (level < g_min_level)
        return;
    const char *level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    char time_buf[32];
    time_t t = time(NULL);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", localtime(&t));

    fprintf(g_log_file, "[%s] [%s] %s:%d: ", time_buf, level_str[level], file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(g_log_file, fmt, args);
    va_end(args);
    fprintf(g_log_file, "\n");
    fflush(g_log_file);
}

// 指桑骂槐宏：记录警告但继续执行
#define LOG_WARN(fmt, ...) log_message(LOG_WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) log_message(LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define LOG_INFO(fmt, ...) log_message(LOG_INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// 条件断言：不终止程序，只记录（指桑骂槐的柔化版）
#define SOFT_ASSERT(expr, msg)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            LOG_WARN("断言失败: %s, 表达式: %s", msg, #expr);                                                          \
        }                                                                                                              \
    } while (0)

// ========== 模拟配置解析器（指桑骂槐示例） ==========
typedef struct
{
    int timeout;
    int retries;
    char server[64];
} Config;

int parse_config(const char *filename, Config *cfg)
{
    if (!filename || !cfg)
    {
        LOG_ERROR("参数无效: filename=%p, cfg=%p", (void *)filename, (void *)cfg);
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        LOG_ERROR("无法打开配置文件 '%s': %s", filename, strerror(errno));
        return -1;
    }

    // 默认值
    cfg->timeout = 30;
    cfg->retries = 3;
    strcpy(cfg->server, "default.com");

    char line[128];
    int line_num = 0;
    while (fgets(line, sizeof(line), fp))
    {
        line_num++;
        char key[64], val[64];
        if (sscanf(line, "%63[^=]=%63s", key, val) != 2)
        {
            // 指桑骂槐：格式错误只记录警告，不停止解析
            LOG_WARN("第%d行格式错误，跳过: %s", line_num, line);
            continue;
        }
        if (strcmp(key, "timeout") == 0)
        {
            int t = atoi(val);
            SOFT_ASSERT(t > 0, "timeout 必须为正数，使用默认值"); // 指桑骂槐
            if (t > 0)
                cfg->timeout = t;
        }
        else if (strcmp(key, "retries") == 0)
        {
            int r = atoi(val);
            SOFT_ASSERT(r >= 0 && r <= 10, "retries 超出范围[0,10]，使用默认值");
            if (r >= 0 && r <= 10)
                cfg->retries = r;
        }
        else if (strcmp(key, "server") == 0)
        {
            strncpy(cfg->server, val, sizeof(cfg->server) - 1);
        }
        else
        {
            LOG_WARN("未知配置项 '%s'，忽略", key);
        }
    }
    fclose(fp);

    LOG_INFO("配置加载成功: timeout=%d, retries=%d, server=%s", cfg->timeout, cfg->retries, cfg->server);
    return 0;
}

// 主函数演示
int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 初始化日志：输出到文件和控制台，最低级别 WARN
    log_init(LOG_WARN, "app.log");

    printf("指桑骂槐示例：尝试解析配置文件\n");
    Config cfg;
    int ret = parse_config("nonexist.conf", &cfg);
    if (ret != 0)
    {
        LOG_ERROR("配置文件缺失，使用硬编码默认值继续运行");
        cfg.timeout = 60;
        cfg.retries = 5;
        strcpy(cfg.server, "fallback.com");
    }

    printf("最终配置: timeout=%d, retries=%d, server=%s\n", cfg.timeout, cfg.retries, cfg.server);

    // 模拟业务运行
    printf("程序继续执行其他任务...\n");

    log_close();
    return 0;
}
