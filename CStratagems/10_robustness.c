/**
 * @file 10_robustness.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 李代桃僵: 错误处理和程序健壮性的平衡
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 10_robustness.exe 10_robustness.c
 * clang -o 10_robustness.exe 10_robustness.c
 *
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>
#include <windows.h>

// 配置结构体
typedef struct
{
    int timeout_ms;
    int max_connections;
    bool enable_cache;
    char log_path[256];
} AppConfig;

// 默认配置（李代桃僵的“僵”）
static const AppConfig DEFAULT_CONFIG = {
    .timeout_ms = 5000, .max_connections = 10, .enable_cache = true, .log_path = "./default.log"};

// 辅助：释放资源的 cleanup 属性（GCC/Clang）
static inline void freep(void *p)
{
    free(*(void **)p);
}

// 尝试从文件加载配置，成功返回true，失败返回false
static bool load_config_from_file(const char *filename, AppConfig *out)
{
    if (!filename || !out)
        return false;

    FILE *fp = fopen(filename, "r");
    if (!fp)
        return false;

    // 使用 fscanf 简单解析（实际应更健壮）
    int timeout, conn, cache;
    char path[256];
    int matched = fscanf(fp, "timeout=%d\nmax_conn=%d\ncache=%d\nlog=%255s\n", &timeout, &conn, &cache, path);
    fclose(fp);

    if (matched != 4)
        return false;

    out->timeout_ms = timeout;
    out->max_connections = conn;
    out->enable_cache = (cache != 0);
    strncpy(out->log_path, path, sizeof(out->log_path) - 1);
    out->log_path[sizeof(out->log_path) - 1] = '\0';
    return true;
}

// 记录错误（李代桃僵中的“李”被替代，但需留痕）
static void log_fallback(const char *reason)
{
    fprintf(stderr, "[警告] 配置加载失败: %s，使用内置默认值\n", reason);
    // 实际可写入系统日志等
}

// 主初始化函数：若配置加载失败，默默替换为默认值（核心：李代桃僵）
AppConfig init_app_config(const char *cfg_file)
{
    AppConfig cfg;
    if (load_config_from_file(cfg_file, &cfg))
    {
        printf("[信息] 成功加载配置文件: %s\n", cfg_file);
        return cfg;
    }
    else
    {
        log_fallback(cfg_file ? cfg_file : "NULL");
        return DEFAULT_CONFIG; // 李代桃僵：用默认值顶替
    }
}

// 模拟程序核心逻辑（使用配置，不关心配置来源）
void run_application(AppConfig cfg)
{
    printf("运行应用: 超时=%dms, 最大连接=%d, 缓存=%s, 日志=%s\n", cfg.timeout_ms, cfg.max_connections,
           cfg.enable_cache ? "启用" : "禁用", cfg.log_path);
    // ... 实际业务
}

// 带有重试机制的网络请求（李代桃僵：重试3次后返回默认响应）
typedef struct
{
    int data;
    bool success;
} Result;

static Result mock_network_call(int attempt)
{
    // 模拟失败
    return (Result){.data = 0, .success = false};
}

Result request_with_retry(int max_retries)
{
    for (int i = 0; i < max_retries; ++i)
    {
        Result res = mock_network_call(i);
        if (res.success)
            return res;
        fprintf(stderr, "重试 %d/%d 失败\n", i + 1, max_retries);
    }
    // 李代桃僵：返回一个安全的空结果而不是崩溃
    return (Result){.data = -1, .success = false};
}

// 现代C: 使用 _Generic 做类型安全的降级获取
#define GET_OR_DEFAULT(value, default_val)                                                                             \
    _Generic((value), int: (value) ? (value) : (default_val), const char *: (value) ? (value) : (default_val))

int main(int argc, char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    // 李代桃僵示例1：配置文件缺失时使用默认值
    const char *cfg_path = (argc > 1) ? argv[1] : "app.conf";
    AppConfig config = init_app_config(cfg_path);
    run_application(config);

    // 示例2：资源分配失败时的替换（使用cleanup自动释放）
    int *p __attribute__((cleanup(freep))) = NULL;
    p = (int *)malloc(sizeof(int));
    if (!p)
    {
        fprintf(stderr, "内存分配失败，使用栈上后备值\n");
        int fallback = 42;
        printf("后备值: %d\n", fallback);
    }
    else
    {
        *p = 100;
        printf("分配成功: %d\n", *p);
    }

    // 示例3：空指针安全访问（返回默认）
    const char *maybe_null = NULL;
    const char *safe_str = maybe_null ? maybe_null : "(null)";
    printf("安全字符串: %s\n", safe_str);

    // 示例4：除零保护（使用三元运算符替换）
    int a = 10, b = 0;
    int quotient = (b != 0) ? (a / b) : 0; // 李代桃僵：除零时返回0
    printf("除法结果（保护后）: %d\n", quotient);

    return 0;
}
