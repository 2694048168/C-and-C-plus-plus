/**
 * @file 12_debug_code.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 打草惊蛇: 调试与错误排查的技巧
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 12_debug_code.exe 12_debug_code.c
 * clang -o 12_debug_code.exe 12_debug_code.c
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifdef _DEBUG
#define DEBUG_LOG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...) ((void)0)
#endif

// 模拟一个带有隐藏越界的函数（“蛇”藏其中）
void process_buffer(int *buf, size_t len)
{
    // 故意：当 len 为 0 时，下面的循环会访问 buf[-1]？不，这里演示另一种越界
    // 实际错误：当 len 为 0 时，循环条件 i <= len-1 变成 i <= -1，导致 i 从 0 开始但条件为假，安全？
    // 换个明显错误：访问 buf[len]（越界1个元素）
    for (size_t i = 0; i <= len; ++i)
    {                    // 错误：应该是 i < len
        buf[i] = (int)i; // 当 i == len 时越界
        DEBUG_LOG("写入 buf[%zu] = %d", i, (int)i);
    }
}

// 使用断言来“打草惊蛇”的包装函数
void safe_process_buffer(int *buf, size_t len)
{
    assert(buf != NULL); // 惊蛇1：空指针
    assert(len > 0);     // 惊蛇2：长度必须大于0
    // 运行时边界检查（可选的防御）
    for (size_t i = 0; i < len; ++i)
    {
        buf[i] = (int)i;
    }
    DEBUG_LOG("安全处理完成，len=%zu", len);
}

// 演示 AddressSanitizer 能捕获的错误（需编译时启用 -fsanitize=address）
void asan_demo(void)
{
    int *arr = (int *)malloc(5 * sizeof(int));
    free(arr);
    // arr[0] = 42;   // 若取消注释，ASan 会报告 use-after-free
    DEBUG_LOG("ASan 演示（需编译时开启）");
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    printf("===== 打草惊蛇调试技巧演示 =====\n");

    // 技巧1：静态断言（编译期检查）
    static_assert(sizeof(int) >= 4, "int 必须至少4字节，否则代码逻辑需调整");

    // 技巧2：运行时断言（调试模式）
    int test_arr[5];
    // 故意传递错误长度，触发断言（调试时程序会中止并显示位置）
    // safe_process_buffer(test_arr, 0);   // 取消注释会触发断言失败

    // 技巧3：使用 DEBUG_LOG 跟踪执行流
    int *heap_buf = (int *)malloc(5 * sizeof(int));
    if (!heap_buf)
    {
        fprintf(stderr, "内存分配失败\n");
        return EXIT_FAILURE;
    }
    process_buffer(heap_buf, 5); // 正确长度，但内部有越界？这里没有，因为 i <= len 当 len=5 会访问索引5
    DEBUG_LOG("heap_buf 处理完成，最后元素=%d", heap_buf[4]);
    free(heap_buf);

    // 技巧4：使用 valgrind 或 ASan 捕捉内存错误（编译命令见后）
    asan_demo();

    // 技巧5：使用 __builtin_trap 制造崩溃点（GCC/Clang）
    // __builtin_trap();   // 取消注释会直接触发 SIGILL，用于强制崩溃调试

    // 技巧6：利用编译器警告（-Wall -Wextra）在编译时惊蛇
    // 例如： int x; printf("%d", x);  // 警告：未初始化

    printf("\n调试核心原则：\n");
    printf("- 使用 assert 捕捉不应发生的条件\n");
    printf("- 日志宏 + 文件行号，快速定位\n");
    printf("- 编译时开启 -fsanitize=address,undefined 动态检查\n");
    printf("- 静态分析工具（clang-analyzer, cppcheck）\n");
    printf("- 单元测试 + 边界值测试\n");

    return 0;
}
