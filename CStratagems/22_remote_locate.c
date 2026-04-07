/**
 * @file 22_remote_locate.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 远交近攻: 模块间的协作与通信策略
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 22_remote_locate.exe 22_remote_locate.c
 * clang -o 22_remote_locate.exe 22_remote_locate.c
 *
 */

#include <Windows.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ========== 远交：远程日志接口（间接调用） ==========
// 定义远程日志发送函数类型（由用户提供具体实现，如UDP、HTTP）
typedef void (*RemoteLogSendFunc)(const char *msg, void *userdata);

// 远程日志器结构（不关心具体发送方式）
typedef struct
{
    RemoteLogSendFunc send; // 远交：通过回调与远端通信
    void *userdata;         // 用户上下文（如socket句柄）
    int enabled;
} RemoteLogger;

// 初始化远程日志器（用户传入发送函数）
void remote_logger_init(RemoteLogger *r, RemoteLogSendFunc send, void *userdata)
{
    r->send = send;
    r->userdata = userdata;
    r->enabled = 1;
}

// 远程写日志（只负责格式化，通过回调“远交”）
void remote_log_write(RemoteLogger *r, const char *level, const char *fmt, ...)
{
    if (!r->enabled || !r->send)
        return;
    char time_buf[32];
    time_t t = time(NULL);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", localtime(&t));

    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    char full[1152];
    snprintf(full, sizeof(full), "[%s] [%s] %s", time_buf, level, msg);
    r->send(full, r->userdata); // 近处只负责传递，远处发送由回调实现
}

// ========== 近攻：本地日志器（直接写文件） ==========
typedef struct
{
    FILE *fp;
    int enabled;
} LocalLogger;

void local_logger_init(LocalLogger *l, const char *filename)
{
    l->fp = fopen(filename, "a");
    l->enabled = (l->fp != NULL);
}

void local_log_write(LocalLogger *l, const char *level, const char *fmt, ...)
{
    if (!l->enabled || !l->fp)
        return;
    char time_buf[32];
    time_t t = time(NULL);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", localtime(&t));

    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    fprintf(l->fp, "[%s] [%s] %s\n", time_buf, level, msg);
    fflush(l->fp);
}

void local_logger_close(LocalLogger *l)
{
    if (l->fp)
        fclose(l->fp);
    l->enabled = 0;
}

// ========== 远端模拟实现（用户提供的“远交”函数） ==========
void mock_remote_send(const char *msg, void *userdata)
{
    // 模拟通过网络发送（比如打印到控制台并显示“远程”）
    printf("[远程发送] %s\n", msg);
    // 实际中可调用 UDP sendto / HTTP POST 等
}

// ========== 近攻直接调用示例：高性能计算模块 ==========
// 一个本地数学库（直接调用，低延迟）
double fast_math_sqrt(double x)
{
    // 近攻：直接计算，不经过任何间接层
    return x * x; // 模拟，实际应为 sqrt
}

// ========== 主程序：展示远交近攻策略 ==========
int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 1. 近攻：本地日志器（直接写文件）
    LocalLogger local;
    local_logger_init(&local, "app_local.log");
    local_log_write(&local, "INFO", "本地日志：程序启动");

    // 2. 远交：远程日志器（通过回调，解耦网络实现）
    RemoteLogger remote;
    remote_logger_init(&remote, mock_remote_send, NULL);
    remote_log_write(&remote, "WARN", "远程日志：磁盘空间不足（模拟）");

    // 3. 近攻：直接调用本地数学函数
    double result = fast_math_sqrt(5.0);
    local_log_write(&local, "DEBUG", "sqrt(5) ≈ %.2f", result);

    // 4. 远交：可以随时更换远程发送策略（如改为UDP）
    // 这里演示切换另一种远程发送函数
    void udp_send(const char *msg, void *userdata)
    {
        printf("[UDP发送] %s\n", msg);
    }
    remote_logger_init(&remote, udp_send, NULL);
    remote_log_write(&remote, "ERROR", "改用UDP发送错误信息");

    // 清理
    local_logger_close(&local);

    printf("\n远交近攻策略总结：\n");
    printf("- 近攻（同一进程）：直接函数调用、共享内存，追求高效\n");
    printf("- 远交（跨模块/网络）：回调、消息队列、RPC，追求解耦和稳定性\n");
    printf("- 现代C语言通过函数指针、线程安全队列实现灵活切换\n");
    return 0;
}
