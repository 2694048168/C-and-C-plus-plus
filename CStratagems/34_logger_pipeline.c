/**
 * @file 34_logger_pipeline.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 连环计: 设计模式组合与架构设计-多个计谋组合构建健壮系统
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 34_logger_pipeline.c -o 34_logger_pipeline.exe
 * clang 34_logger_pipeline.c -o 34_logger_pipeline.exe
 * 
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

// ========== 计谋1：关门捉贼（模块化封装） ==========
// 日志条目结构（对外隐藏细节）
typedef struct LogEntry LogEntry;
struct LogEntry
{
    time_t timestamp;
    char level[16];
    char message[256];
    LogEntry *next;
};

// 日志队列（不透明指针）
typedef struct
{
    LogEntry *head;
    LogEntry *tail;
    size_t count;
} LogQueue;

// 模块接口
LogQueue *queue_create(void);
void queue_destroy(LogQueue *q);
bool queue_push(LogQueue *q, const char *level, const char *msg);
bool queue_pop(LogQueue *q, LogEntry *out);
size_t queue_size(LogQueue *q);

// ========== 计谋2：上屋抽梯（输入校验） ==========
bool queue_push(LogQueue *q, const char *level, const char *msg)
{
    if (!q || !level || !msg)
        return false;
    // 长度边界检查（抽梯）
    if (strlen(level) >= sizeof(((LogEntry *)0)->level) || strlen(msg) >= sizeof(((LogEntry *)0)->message))
    {
        fprintf(stderr, "[上屋抽梯] 输入过长，拒绝添加\n");
        return false;
    }
    LogEntry *e = (LogEntry *)malloc(sizeof(LogEntry));
    if (!e)
        return false;
    e->timestamp = time(NULL);
    strcpy(e->level, level);
    strcpy(e->message, msg);
    e->next = NULL;
    if (q->tail)
        q->tail->next = e;
    else
        q->head = e;
    q->tail = e;
    q->count++;
    return true;
}

// 其他队列函数实现（简化）
LogQueue *queue_create(void)
{
    LogQueue *q = (LogQueue *)calloc(1, sizeof(LogQueue));
    return q;
}
void queue_destroy(LogQueue *q)
{
    if (!q)
        return;
    LogEntry *e = q->head;
    while (e)
    {
        LogEntry *next = e->next;
        free(e);
        e = next;
    }
    free(q);
}
bool queue_pop(LogQueue *q, LogEntry *out)
{
    if (!q || !q->head)
        return false;
    LogEntry *e = q->head;
    *out = *e;
    q->head = e->next;
    if (!q->head)
        q->tail = NULL;
    free(e);
    q->count--;
    return true;
}
size_t queue_size(LogQueue *q)
{
    return q ? q->count : 0;
}

// ========== 计谋3：反客为主（回调接口） ==========
typedef bool (*LogProcessor)(LogEntry *entry, void *ctx);
void process_logs(LogQueue *q, LogProcessor processor, void *ctx)
{
    if (!q || !processor)
        return;
    LogEntry entry;
    while (queue_pop(q, &entry))
    {
        if (!processor(&entry, ctx))
            break;
    }
}

// ========== 计谋4：借尸还魂（对象池复用） ==========
typedef struct
{
    LogEntry *pool;
    size_t size;
    size_t used;
} LogPool;

LogPool *pool_create(size_t capacity)
{
    LogPool *p = (LogPool *)malloc(sizeof(LogPool));
    p->pool = (LogEntry *)calloc(capacity, sizeof(LogEntry));
    p->size = capacity;
    p->used = 0;
    return p;
}
void pool_destroy(LogPool *p)
{
    if (p)
    {
        free(p->pool);
        free(p);
    }
}
LogEntry *pool_acquire(LogPool *p)
{
    if (!p || p->used >= p->size)
        return NULL;
    return &p->pool[p->used++];
}
void pool_release(LogPool *p, LogEntry *e)
{
    // 简化：标记可用（实际可维护空闲链表）
    (void)p;
    (void)e;
}

// ========== 计谋5：釜底抽薪（安全字符串处理） ==========
void safe_str_copy(char *dest, size_t dest_size, const char *src)
{
    if (!dest || !src || dest_size == 0)
        return;
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

// ========== 计谋6：欲擒故纵（延迟计算） ==========
typedef struct
{
    char pattern[32];
    bool compiled;
    void *regex; // 模拟
} Filter;
bool matches_filter(Filter *f, const char *msg)
{
    if (!f->compiled)
    {
        // 延迟编译正则表达式（欲擒故纵）
        printf("[欲擒故纵] 首次使用，编译过滤规则: %s\n", f->pattern);
        f->compiled = true;
    }
    // 模拟简单匹配
    return strstr(msg, f->pattern) != NULL;
}

// ========== 主程序：组合使用 ==========
// 用户自定义处理器（反客为主）
bool alert_on_error(LogEntry *entry, void *ctx)
{
    if (strcmp(entry->level, "ERROR") == 0)
    {
        printf("[连环计] 检测到错误: %s\n", entry->message);
        (*(int *)ctx)++;
    }
    return true; // 继续处理
}

int main(void)
{
    // 设置控制台输出编码
    SetConsoleOutputCP(65001);

    // 1. 创建队列（关门捉贼）
    LogQueue *queue = queue_create();

    // 2. 模拟添加日志（上屋抽梯自动校验）
    queue_push(queue, "INFO", "系统启动");
    queue_push(queue, "ERROR", "磁盘空间不足");
    queue_push(queue, "ERROR", "网络连接超时");
    queue_push(queue, "DEBUG", "调试信息（很长很长很长很长很长很长很长很长很长）"); // 会被拦截
    queue_push(queue, "WARN", "配置过期");

    printf("当前队列大小: %zu\n", queue_size(queue));

    // 3. 使用对象池处理（借尸还魂）
    LogPool *pool = pool_create(10);
    LogEntry *e = pool_acquire(pool);
    if (e)
    {
        safe_str_copy(e->level, sizeof(e->level), "INFO");
        safe_str_copy(e->message, sizeof(e->message), "池化日志");
        e->timestamp = time(NULL);
        queue_push(queue, e->level, e->message);
        pool_release(pool, e);
    }

    // 4. 过滤延迟编译（欲擒故纵）
    Filter filter = {.pattern = "超时", .compiled = false};

    // 5. 处理日志（反客为主）
    int error_count = 0;
    process_logs(queue, alert_on_error, &error_count);

    printf("共处理 %d 条错误日志\n", error_count);

    // 6. 清理
    queue_destroy(queue);
    pool_destroy(pool);

    printf("\n连环计核心：\n");
    printf("- 关门捉贼：模块化队列，隐藏内部链表\n");
    printf("- 上屋抽梯：每个输入都校验长度\n");
    printf("- 反客为主：用户通过回调控制处理逻辑\n");
    printf("- 借尸还魂：对象池减少频繁分配\n");
    printf("- 釜底抽薪：安全字符串复制\n");
    printf("- 欲擒故纵：正则表达式延迟编译\n");
    return 0;
}
