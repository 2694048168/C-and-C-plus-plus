/**
 * @file 30_API_cfg.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 美人计: 接口设计-简洁优雅的接口隐藏复杂实现
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 30_API_cfg.c -o 30_API_cfg.exe
 * clang 30_API_cfg.c -o 30_API_cfg.exe
 *
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// ========== 美人计：对外接口简洁优雅 ==========
// 用户只需要看到这几个函数
void cfg_load(const char *filename);
int cfg_get_int(const char *key);
const char *cfg_get_str(const char *key);
void cfg_unload(void);

// ========== 内部实现（复杂，但对用户隐藏） ==========
typedef struct ConfigEntry
{
    char *key;
    char *value;
    struct ConfigEntry *next;
} ConfigEntry;

static ConfigEntry *g_config = NULL;

// 内部函数：去除字符串首尾空格
static char *trim(char *str)
{
    while (isspace((unsigned char)*str))
        str++;
    if (*str == 0)
        return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end))
        end--;
    end[1] = '\0';
    return str;
}

// 内部函数：解析一行 "key = value"
static void parse_line(char *line)
{
    char *eq = strchr(line, '=');
    if (!eq)
        return;
    *eq = '\0';
    char *key = trim(line);
    char *val = trim(eq + 1);
    if (strlen(key) == 0)
        return;

    ConfigEntry *entry = malloc(sizeof(ConfigEntry));
    entry->key = strdup(key);
    entry->value = strdup(val);
    entry->next = g_config;
    g_config = entry;
}

// 内部函数：查找条目
static ConfigEntry *find_entry(const char *key)
{
    for (ConfigEntry *e = g_config; e; e = e->next)
    {
        if (strcmp(e->key, key) == 0)
            return e;
    }
    return NULL;
}

// 公开接口：加载配置文件（内部做文件打开、解析、错误处理）
void cfg_load(const char *filename)
{
    // 清理旧配置
    cfg_unload();

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "警告: 无法打开配置文件 %s，使用默认值\n", filename);
        return;
    }

    char line[512];
    while (fgets(line, sizeof(line), fp))
    {
        // 去除换行符
        line[strcspn(line, "\n")] = '\0';
        // 跳过空行和注释
        if (line[0] == '#' || line[0] == '\0')
            continue;
        parse_line(line);
    }
    fclose(fp);
}

// 公开接口：获取整数值（内部处理不存在、非数字等情况）
int cfg_get_int(const char *key)
{
    ConfigEntry *e = find_entry(key);
    if (!e)
    {
        fprintf(stderr, "警告: 配置键 '%s' 不存在，返回默认值 0\n", key);
        return 0;
    }
    char *endptr;
    long val = strtol(e->value, &endptr, 10);
    if (*endptr != '\0')
    {
        fprintf(stderr, "警告: 配置键 '%s' 的值 '%s' 不是整数，返回 0\n", key, e->value);
        return 0;
    }
    return (int)val;
}

// 公开接口：获取字符串值
const char *cfg_get_str(const char *key)
{
    ConfigEntry *e = find_entry(key);
    if (!e)
    {
        fprintf(stderr, "警告: 配置键 '%s' 不存在，返回空字符串\n", key);
        return "";
    }
    return e->value;
}

// 公开接口：释放资源
void cfg_unload(void)
{
    ConfigEntry *e = g_config;
    while (e)
    {
        ConfigEntry *next = e->next;
        free(e->key);
        free(e->value);
        free(e);
        e = next;
    }
    g_config = NULL;
}

// ========== 主程序：用户只看到简单接口 ==========
int main(void)
{
    SetConsoleOutputCP(65001);

    // 用户只需要这几行
    cfg_load("app.conf");
    int timeout = cfg_get_int("timeout");
    const char *name = cfg_get_str("server_name");
    cfg_unload();

    printf("配置读取结果:\n");
    printf("  timeout = %d\n", timeout);
    printf("  server_name = %s\n", name);

    printf("\n美人计核心：\n");
    printf("- 用户只需记住 cfg_load/cfg_get_int/cfg_get_str/cfg_unload\n");
    printf("- 内部实现文件解析、缓存、错误处理、类型转换\n");
    printf("- 即使配置文件缺失或格式错误，接口仍优雅返回默认值\n");
    printf("- 复杂逻辑被“美人”外貌隐藏，降低使用门槛\n");

    return 0;
}
