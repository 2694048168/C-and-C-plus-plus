/**
 * @file 20_free_resource.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 金蝉脱壳: 资源释放和程序退出(释放内存/关闭文件/子线程正确退出)
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 20_free_resource.c -o 20_free_resource.exe
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// ========== 金蝉脱壳：资源记录与清理 ==========
typedef struct
{
    void *ptr;                    // 动态分配的内存
    FILE *fp;                     // 打开的文件
    HANDLE hThread;               // 子线程句柄
    volatile BOOL thread_running; // 线程运行标志
} AppResources;

static AppResources g_res = {NULL, NULL, NULL, FALSE};

// 清理函数1：释放内存
static void cleanup_memory(void)
{
    if (g_res.ptr)
    {
        free(g_res.ptr);
        g_res.ptr = NULL;
        printf("[清理] 内存已释放\n");
    }
}

// 清理函数2：关闭文件
static void cleanup_file(void)
{
    if (g_res.fp)
    {
        fclose(g_res.fp);
        g_res.fp = NULL;
        printf("[清理] 文件已关闭\n");
    }
}

// 清理函数3：等待子线程退出并关闭句柄
static void cleanup_thread(void)
{
    if (g_res.hThread)
    {
        if (g_res.thread_running)
        {
            g_res.thread_running = FALSE;                 // 通知线程退出
            WaitForSingleObject(g_res.hThread, INFINITE); // 等待线程结束
        }
        CloseHandle(g_res.hThread);
        g_res.hThread = NULL;
        printf("[清理] 子线程已正确退出，句柄已关闭\n");
    }
}

// 统一的退出清理函数（通过 atexit 注册）
static void global_cleanup(void)
{
    printf("\n[金蝉脱壳] 程序退出，开始清理资源...\n");
    cleanup_thread();
    cleanup_file();
    cleanup_memory();
    printf("[金蝉脱壳] 所有资源已释放，程序正常退出\n");
}

// ========== 子线程函数（Windows API 签名） ==========
DWORD WINAPI worker_thread(LPVOID lpParam)
{
    printf("[线程] 子线程启动，每隔1秒工作一次\n");
    while (g_res.thread_running)
    {
        printf("[线程] 工作中...\n");
        Sleep(1000); // 睡眠1秒
    }
    printf("[线程] 收到退出信号，子线程结束\n");
    return 0;
}

// ========== 初始化资源（可能部分失败） ==========
int init_resources(void)
{
    // 分配内存
    g_res.ptr = malloc(1024);
    if (!g_res.ptr)
    {
        printf("malloc 失败\n");
        return -1;
    }
    printf("[资源] 分配了 1024 字节内存\n");

    // 打开文件
    g_res.fp = fopen("test.log", "w");
    if (!g_res.fp)
    {
        printf("fopen 失败\n");
        return -1;
    }
    printf("[资源] 打开文件 test.log\n");

    // 创建子线程
    g_res.thread_running = TRUE;
    g_res.hThread = CreateThread(NULL,          // 安全属性
                                 0,             // 栈大小（0=默认）
                                 worker_thread, // 线程函数
                                 NULL,          // 参数
                                 0,             // 创建标志
                                 NULL           // 线程ID（可选）
    );
    if (!g_res.hThread)
    {
        printf("CreateThread 失败，错误码: %lu\n", GetLastError());
        g_res.thread_running = FALSE;
        return -1;
    }
    printf("[资源] 创建子线程成功\n");
    return 0;
}

// 模拟中途出错（仅演示）
void simulate_partial_failure(void)
{
    printf("\n[模拟] 尝试打开一个不存在的文件（故意出错）\n");
    FILE *bad = fopen("C:\\nonexistent\\file.txt", "r");
    if (!bad)
    {
        printf("[错误] 打开文件失败: %s\n", "文件不存在");
        // 此时不需要手动清理，因为 atexit 会处理
    }
}

int main(void)
{
    // 设置控制台代码页为 UTF-8（避免中文乱码）
    SetConsoleOutputCP(CP_UTF8);

    // 注册退出清理函数（无论 exit 或 main 返回都会调用）
    if (atexit(global_cleanup) != 0)
    {
        fprintf(stderr, "atexit 注册失败\n");
        return EXIT_FAILURE;
    }

    // 初始化资源
    if (init_resources() != 0)
    {
        fprintf(stderr, "资源初始化失败，程序即将退出\n");
        return EXIT_FAILURE;
    }

    // 主线程工作一段时间
    printf("\n[主线程] 开始工作，5 秒后退出...\n");
    for (int i = 1; i <= 5; ++i)
    {
        printf("[主线程] 工作 %d/5\n", i);
        Sleep(1000);
    }

    simulate_partial_failure();

    printf("\n[主线程] 工作完成，即将退出\n");
    // main 返回时自动调用 global_cleanup，实现金蝉脱壳
    return EXIT_SUCCESS;
}
