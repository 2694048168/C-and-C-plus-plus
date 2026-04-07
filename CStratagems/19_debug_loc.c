/**
 * @file 19_debug_loc.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 浑水摸鱼: 利用混乱局面,寻找解决问题的机会(利用错误信息/调试工具定位问题)
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 19_debug_loc.exe 19_debug_loc.c
 * clang -o 19_debug_loc.exe 19_debug_loc.c
 * 
 */

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// 调试宏：浑水摸鱼时留下线索
#ifdef DEBUG
#define LOG_MSG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_MSG(fmt, ...) ((void)0)
#endif

// 模拟一个混乱的解析函数：可能产生多种错误
bool parse_config(const char *filename, int *timeout, bool *enabled)
{
    if (!filename || !timeout || !enabled)
    {
        LOG_MSG("空指针传入");
        return false;
    }

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        LOG_MSG("打开文件失败: %s, errno=%d (%s)", filename, errno, strerror(errno));
        return false;
    }

    // 模拟读取配置（可能格式错误）
    char line[64];
    if (!fgets(line, sizeof(line), fp))
    {
        LOG_MSG("读取文件为空或失败");
        fclose(fp);
        return false;
    }

    // 解析 "timeout=xxx" 和 "enabled=1/0"
    int parsed_timeout = -1;
    int parsed_enabled = -1;
    if (sscanf(line, "timeout=%d enabled=%d", &parsed_timeout, &parsed_enabled) != 2)
    {
        LOG_MSG("配置格式错误: 期望 'timeout=数字 enabled=数字', 实际得到 '%s'", line);
        fclose(fp);
        return false;
    }

    *timeout = parsed_timeout;
    *enabled = (parsed_enabled != 0);
    fclose(fp);
    LOG_MSG("解析成功: timeout=%d, enabled=%d", *timeout, *enabled);
    return true;
}

// 故意制造一个缓冲区溢出（仅在调试版本用 assert 捕获）
void dangerous_copy(const char *src)
{
    char buf[5];
    // 浑水摸鱼：用断言检查长度，避免真正溢出
    assert(strlen(src) < sizeof(buf));
    strcpy(buf, src);
    LOG_MSG("复制成功: %s", buf);
}

// 模拟利用错误信息进行“摸鱼”恢复
void chaotic_function(int choice)
{
    switch (choice)
    {
    case 1: {
        int *p = NULL;
        // 故意解引用空指针，但在调试版本中可以用 assert 捕获
        // 实际中 AddressSanitizer 会给出精确位置
        LOG_MSG("即将访问空指针（模拟崩溃）");
        // assert(p != NULL);  // 如果启用会直接中止
        break;
    }
    case 2: {
        int arr[3] = {1, 2, 3};
        LOG_MSG("即将越界访问 arr[5]");
        // assert(5 < 3);  // 越界检测
        break;
    }
    default:
        LOG_MSG("未知选择，无事发生");
    }
}

int main(int argc, char *argv[])
{
    SetConsoleOutputCP(65001);

    // 浑水摸鱼第一式：从错误返回值中提取信息
    int timeout;
    bool enabled;
    const char *cfg = (argc > 1) ? argv[1] : "nonexist.conf";

    if (!parse_config(cfg, &timeout, &enabled))
    {
        fprintf(stderr, "配置解析失败，使用默认值（浑水摸鱼：从错误中恢复）\n");
        timeout = 30;
        enabled = true;
    }
    printf("最终配置: timeout=%d, enabled=%d\n", timeout, enabled);

    // 第二式：利用断言在调试阶段捕获逻辑错误
    int value = 42;
    assert(value > 0); // 正常通过

    // 第三式：演示字符串复制时的边界检查
    const char *long_str = "HelloWorld";
    // dangerous_copy(long_str);  // 取消注释会触发断言失败，指出问题行

    // 第四式：故意制造混乱，然后利用调试器或 Sanitizer 定位
    chaotic_function(1);

    // 第五式：使用 errno 摸鱼
    FILE *fp = fopen("/nonexistent/file", "r");
    if (!fp)
    {
        LOG_MSG("fopen 失败: %s", strerror(errno));
        perror("perror 直接输出");
    }

    printf("\n浑水摸鱼核心技巧：\n");
    printf("- 使用 LOG_MSG 宏记录文件/行号，追踪混乱现场\n");
    printf("- assert 捕获不应该发生的条件，快速定位逻辑错误\n");
    printf("- 检查 errno 并用 strerror/perror 获得系统错误描述\n");
    printf("- 编译时启用 -fsanitize=address,undefined 让混乱现形\n");
    printf("- 使用调试器 (gdb/lldb) 在崩溃点回溯调用栈\n");

    return 0;
}
