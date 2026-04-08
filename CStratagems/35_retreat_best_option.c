/**
 * @file 35_retreat_best_option.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 走为上计: 优雅撤退-保护程序安全
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 35_retreat_best_option.c -o 35_retreat_best_option.exe
 * clang 35_retreat_best_option.c -o 35_retreat_best_option.exe
 *
 */

#include <setjmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define SIGPIPE 13 // 定义占位，但实际不使用
// Windows 控制台设置 UTF-8（可选）
static void init_console(void)
{
    SetConsoleOutputCP(CP_UTF8);
}
#else
#include <unistd.h>
#define init_console() ((void)0)
#endif

// ========== 资源与状态记录 ==========
typedef struct
{
    FILE *log_fp;
    void *temp_buffer;
    int transaction_active;
    char last_error[256];
} AppContext;

static AppContext g_ctx = {NULL, NULL, 0, ""};
static jmp_buf recovery_env;
static atomic_int g_exit_flag = 0;

// ========== 走为上计：清理函数 ==========
void cleanup_resources(void)
{
    if (g_ctx.log_fp)
    {
        fprintf(g_ctx.log_fp, "[撤退] 程序退出，执行资源清理\n");
        fclose(g_ctx.log_fp);
        g_ctx.log_fp = NULL;
    }
    if (g_ctx.temp_buffer)
    {
        free(g_ctx.temp_buffer);
        g_ctx.temp_buffer = NULL;
    }
    if (g_ctx.transaction_active)
    {
        printf("[回滚] 事务已回滚\n");
        g_ctx.transaction_active = 0;
    }
}

void setup_atexit(void)
{
    atexit(cleanup_resources);
}

// ========== 走为上计：信号处理（优雅退出） ==========
void safe_signal_handler(int sig)
{
    if (atomic_exchange(&g_exit_flag, 1))
        return;

    const char *msg = NULL;
    switch (sig)
    {
    case SIGINT:
        msg = "收到中断信号 (Ctrl+C)";
        break;
    case SIGTERM:
        msg = "收到终止信号";
        break;
#ifdef SIGSEGV
    case SIGSEGV:
        msg = "段错误（程序将退出）";
        break;
#endif
    default:
        msg = "未知信号";
    }
    fprintf(stderr, "[信号] %s，准备撤退\n", msg);
    if (g_ctx.log_fp)
    {
        fprintf(g_ctx.log_fp, "[信号] %s，时间: %ld\n", msg, time(NULL));
    }
    _exit(EXIT_FAILURE);
}

void setup_signals(void)
{
    signal(SIGINT, safe_signal_handler);
    signal(SIGTERM, safe_signal_handler);
#ifdef SIGSEGV
    signal(SIGSEGV, safe_signal_handler);
#endif
#ifndef _WIN32
    // Windows 没有 SIGPIPE，忽略相关设置
    signal(SIGPIPE, SIG_IGN);
#endif
}

// ========== 模拟事务操作 ==========
int start_transaction(void)
{
    if (g_ctx.transaction_active)
    {
        snprintf(g_ctx.last_error, sizeof(g_ctx.last_error), "事务已激活，不能重复开始");
        return -1;
    }
    g_ctx.transaction_active = 1;
    printf("[事务] 开始\n");
    return 0;
}

int commit_transaction(void)
{
    if (!g_ctx.transaction_active)
    {
        snprintf(g_ctx.last_error, sizeof(g_ctx.last_error), "无活跃事务可提交");
        return -1;
    }
    printf("[事务] 提交成功\n");
    g_ctx.transaction_active = 0;
    return 0;
}

void rollback_transaction(void)
{
    if (g_ctx.transaction_active)
    {
        printf("[回滚] 撤销事务更改\n");
        g_ctx.transaction_active = 0;
    }
}

int perform_critical_operation(int value)
{
    if (value < 0)
    {
        snprintf(g_ctx.last_error, sizeof(g_ctx.last_error), "非法参数: %d", value);
        return -1;
    }
    if (value > 1000)
    {
        snprintf(g_ctx.last_error, sizeof(g_ctx.last_error), "数据溢出: %d", value);
        return -1;
    }
    return value * 2;
}

// ========== 主程序 ==========
int main(int argc, char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    init_console();
    setup_atexit();
    setup_signals();

    g_ctx.log_fp = fopen("retreat.log", "a");
    if (!g_ctx.log_fp)
    {
        fprintf(stderr, "无法打开日志文件，撤退\n");
        return EXIT_FAILURE;
    }
    g_ctx.temp_buffer = malloc(1024);
    if (!g_ctx.temp_buffer)
    {
        fprintf(g_ctx.log_fp, "内存分配失败，撤退\n");
        return EXIT_FAILURE;
    }

    if (setjmp(recovery_env) == 0)
    {
        if (start_transaction() != 0)
        {
            longjmp(recovery_env, 1);
        }

        int input = (argc > 1) ? atoi(argv[1]) : 500;
        int result = perform_critical_operation(input);
        if (result == -1)
        {
            fprintf(g_ctx.log_fp, "致命错误: %s\n", g_ctx.last_error);
            longjmp(recovery_env, 1);
        }

        printf("操作成功，结果: %d\n", result);
        if (commit_transaction() != 0)
        {
            longjmp(recovery_env, 1);
        }
    }
    else
    {
        rollback_transaction();
        fprintf(g_ctx.log_fp, "[撤退] 时间: %ld，原因: %s\n", time(NULL), g_ctx.last_error);
        printf("程序已安全撤退，详情见 retreat.log\n");
        return EXIT_FAILURE;
    }

    printf("程序正常结束\n");
    return EXIT_SUCCESS;
}
