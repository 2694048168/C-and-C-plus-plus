/**
 * @file 32_sowing_discord.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 反间计: 利用恶意输入加固程序防御
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 32_sowing_discord.c -o 32_sowing_discord.exe
 * clang 32_sowing_discord.c -o 32_sowing_discord.exe
 *
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

// ========== 反间计：利用恶意输入加固防御 ==========
// 攻击特征库（可动态更新）
#define MAX_PATTERNS 10
static char *sql_patterns[] = {"SELECT", "DROP",   "INSERT", "UPDATE", "DELETE",
                               "OR 1=1", "' OR '", "UNION",  "exec",   "xp_cmdshell"};
static int pattern_count = 10;

// 黑名单结构（IP + 封禁时间）
typedef struct Blacklist
{
    char ip[16];
    time_t expire_time;
    struct Blacklist *next;
} Blacklist;
static Blacklist *g_blacklist = NULL;

// 日志记录（反间计的关键：收集攻击证据）
void log_attack(const char *ip, const char *payload, const char *reason)
{
    time_t now = time(NULL);
    printf("[攻击日志] %s | IP: %s | 载荷: %s | 原因: %s\n", ctime(&now), ip, payload, reason);
    // 实际可写入文件或发送到安全中心
}

// 检查是否包含恶意特征（动态检测）
int contains_malicious(const char *input)
{
    char upper_input[512];
    strncpy(upper_input, input, sizeof(upper_input) - 1);
    upper_input[sizeof(upper_input) - 1] = '\0';
    for (char *p = upper_input; *p; p++)
        *p = toupper(*p);

    for (int i = 0; i < pattern_count; i++)
    {
        if (strstr(upper_input, sql_patterns[i]))
        {
            return 1;
        }
    }
    // 检测缓冲区溢出特征（长字符串或大量重复字符）
    if (strlen(input) > 200)
        return 1;
    return 0;
}

// 动态更新特征库（反间计：从攻击中学习）
void add_pattern(const char *new_pattern)
{
    if (pattern_count >= MAX_PATTERNS)
    {
        printf("[防御] 特征库已满，无法添加新规则\n");
        return;
    }
    sql_patterns[pattern_count++] = strdup(new_pattern);
    printf("[防御] 动态新增攻击特征: %s\n", new_pattern);
}

// 将攻击者加入黑名单（反间计：利用敌方信息反制）
void block_ip(const char *ip, int duration_seconds)
{
    Blacklist *entry = g_blacklist;
    while (entry)
    {
        if (strcmp(entry->ip, ip) == 0)
        {
            entry->expire_time = time(NULL) + duration_seconds;
            printf("[防御] 更新黑名单: %s 封禁至 %ld\n", ip, entry->expire_time);
            return;
        }
        entry = entry->next;
    }
    Blacklist *new_entry = malloc(sizeof(Blacklist));
    strcpy(new_entry->ip, ip);
    new_entry->expire_time = time(NULL) + duration_seconds;
    new_entry->next = g_blacklist;
    g_blacklist = new_entry;
    printf("[防御] 新增黑名单: %s 封禁 %d 秒\n", ip, duration_seconds);
}

// 检查IP是否被封禁
int is_blocked(const char *ip)
{
    time_t now = time(NULL);
    Blacklist *prev = NULL;
    Blacklist *entry = g_blacklist;
    while (entry)
    {
        if (strcmp(entry->ip, ip) == 0)
        {
            if (now < entry->expire_time)
            {
                return 1;
            }
            else
            {
                // 过期移除
                if (prev)
                    prev->next = entry->next;
                else
                    g_blacklist = entry->next;
                free(entry);
                return 0;
            }
        }
        prev = entry;
        entry = entry->next;
    }
    return 0;
}

// 模拟处理请求（反间计主流程）
void handle_request(const char *client_ip, const char *payload)
{
    if (is_blocked(client_ip))
    {
        printf("[拦截] IP %s 已被封禁，拒绝访问\n", client_ip);
        return;
    }

    if (contains_malicious(payload))
    {
        log_attack(client_ip, payload, "检测到SQL注入或溢出特征");
        // 反间计：根据攻击严重程度动态调整封禁时间
        int block_seconds = 60;
        if (strlen(payload) > 300)
            block_seconds = 300;
        block_ip(client_ip, block_seconds);

        // 从攻击载荷中提取新模式（示例：若出现未见过的关键字）
        char upper[256];
        strcpy(upper, payload);
        for (char *p = upper; *p; p++)
            *p = toupper(*p);
        if (strstr(upper, "SLEEP") && pattern_count < MAX_PATTERNS)
        {
            add_pattern("SLEEP");
        }
    }
    else
    {
        printf("[正常] IP %s 请求: %s\n", client_ip, payload);
    }
}

// 清理黑名单（程序退出时）
void cleanup_blacklist(void)
{
    Blacklist *entry = g_blacklist;
    while (entry)
    {
        Blacklist *next = entry->next;
        free(entry);
        entry = next;
    }
}

int main(void)
{
    SetConsoleOutputCP(65001);

    printf("===== 反间计：利用恶意输入加固防御 =====\n");
    printf("模拟请求处理（输入 'quit' 退出）\n\n");

    char ip[16] = "192.168.1.100"; // 模拟攻击源IP
    char input[512];

    while (1)
    {
        printf("\n请输入请求载荷 (或 quit): ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin))
            break;
        input[strcspn(input, "\n")] = '\0';
        if (strcmp(input, "quit") == 0)
            break;

        handle_request(ip, input);
    }

    cleanup_blacklist();
    printf("\n反间计核心思想：\n");
    printf("- 程序主动接收恶意输入，分析攻击特征\n");
    printf("- 利用这些特征动态更新防御规则（特征库、黑名单）\n");
    printf("- 攻击者越尝试，系统防御越强\n");
    printf("- 日志记录可用于溯源和取证\n");
    return 0;
}
