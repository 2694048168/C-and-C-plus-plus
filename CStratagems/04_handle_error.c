/**
 * @file 04_handle_error.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 趁火打劫: 错误处理与异常捕获
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o handle_error 04_handle_error.c
 * clang -o handle_error 04_handle_error.c
 *
 */

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>

// 定义错误码
typedef enum
{
    ERR_OK = 0,
    ERR_FILE_OPEN,
    ERR_MEMORY_ALLOC,
    ERR_FILE_READ,
    ERR_PARSE
} ErrorCode;

// 全局异常环境（每个线程独立，这里简化）
static jmp_buf env;
static volatile ErrorCode g_error = ERR_OK; // volatile 防止 longjmp 优化问题

// 抛异常（相当于“趁火”动作）
noreturn void throw_error(ErrorCode err)
{
    g_error = err;
    longjmp(env, 1); // 跳回 setjmp 点
}

// 模拟文件解析：可能多处出错
void parse_file(const char *filename)
{
    FILE *fp = NULL;
    char *buffer = NULL;
    size_t buffer_size = 0;

    // 步骤1: 打开文件
    fp = fopen(filename, "r");
    if (!fp)
    {
        throw_error(ERR_FILE_OPEN); // 趁火：立即跳转到清理区
    }

    // 步骤2: 分配读取缓冲区
    buffer_size = 1024;
    buffer = (char *)malloc(buffer_size);
    if (!buffer)
    {
        throw_error(ERR_MEMORY_ALLOC);
    }

    // 步骤3: 读取文件内容
    if (fgets(buffer, buffer_size, fp) == NULL)
    {
        throw_error(ERR_FILE_READ);
    }

    // 步骤4: 假装解析（可能失败）
    if (strlen(buffer) == 0)
    {
        throw_error(ERR_PARSE);
    }

    printf("[Success] File Content: %s", buffer);

    // 正常释放（不用跳转）
    free(buffer);
    fclose(fp);
    return;

    // ----- 错误处理区：“趁火打劫” -----
    // 注意：这里不是函数内独立块，而是用标签 + 通过 longjmp 直接跳入
    // 实际 C 语法需要把清理代码放在函数末尾，并通过 goto 或上述 throw 跳转。
    // 为使结构清晰，我们使用 goto 风格改写会更直观。但为了展示 setjmp/longjmp，
    // 下面展示另一种方式：在调用 parse_file 的外层做清理。然而资源清理通常应在函数内完成。
    // 更好的演示：在函数内使用 setjmp? 不，通常是调用者设置跳转点，被调函数负责释放自己的资源。
    // 这里为了展示“趁火打劫”的精髓，重构为：函数内若出错，先自己部分清理，再 longjmp。
    // 但上面代码中，如果 longjmp 直接跳走，已经分配的 buffer 或 fp 无法释放。所以我们需要在 throw 前先清理。
}

// 改进版本：将“趁火打劫”内嵌到每个错误发生点之前（手动清理）
// 这才是正确的 C 风格异常模拟：抛出前清理已获得的资源。
void parse_file_v2(const char *filename)
{
    FILE *fp = NULL;
    char *buffer = NULL;

    fp = fopen(filename, "r");
    if (!fp)
    {
        throw_error(ERR_FILE_OPEN);
    }

    buffer = (char *)malloc(1024);
    if (!buffer)
    {
        fclose(fp); // 趁火：清理已打开的文件
        throw_error(ERR_MEMORY_ALLOC);
    }

    if (fgets(buffer, 1024, fp) == NULL)
    {
        free(buffer); // 趁火：释放内存
        fclose(fp);
        throw_error(ERR_FILE_READ);
    }

    // 正常路径
    printf("内容: %s", buffer);
    free(buffer);
    fclose(fp);
}

int main(void)
{
    // 设置跳转点（类似 try 块）
    if (setjmp(env) == 0)
    {
        // 正常执行区
        parse_file_v2("test.txt");
    }
    else
    {
        // 异常捕获区 —— “趁火打劫”的主战场
        switch (g_error)
        {
        case ERR_FILE_OPEN:
            fprintf(stderr, "[hijack] Error: Unable to open the file. Log has been recorded\n");
            break;
        case ERR_MEMORY_ALLOC:
            fprintf(stderr, "[hijack] Error: Out of memory, some resources have been cleared\n");
            break;
        case ERR_FILE_READ:
            fprintf(stderr, "[hijack] Error: File read failure, buffer has been released\n");
            break;
        case ERR_PARSE:
            fprintf(stderr, "[hijack] Error: Parsing failed, status rolled back\n");
            break;
        default:
            fprintf(stderr, "[hijack] Unknown error, terminating program\n");
        }
        // 可在此执行额外全局清理，比如关闭数据库连接、写错误日志等
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
