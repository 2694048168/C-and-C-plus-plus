/**
 * @file MemoryLeak.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#if __linux__
#    define _GUN_SOURCE
#    include <dlfcn.h>
#    include <unistd.h>

#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
// #include <iostream>

// 单文件-内存泄漏检测
#if 1
void *_malloc(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);

    // ======== log-file way ========
    char buff[128] = {0};
    sprintf(buff, "./mem/%p.mem", ptr);

    FILE *fp = fopen(buff, "w");
    // fprintf(fp, "[+%s:%d] ---> addr:%lld\n", file, line, size);
    fprintf(fp, "[+%s:%d] ---> addr:%ld\n", file, line, size);
    fflush(fp);
    fclose(fp);

    // ======== terminal-console way ========
    // printf("[+%s:%d], memory-bytes size: %lld\n", file, line, size);
    printf("[+%s:%d], memory-bytes size: %ld\n", file, line, size);

    return ptr;
}

void _free(void *ptr, const char *file, int line)
{
    // ======== log-file way ========
    char buff[128] = {0};
    sprintf(buff, "./mem/%p.mem", ptr);

    if (unlink(buff) < 0) /* double free */
    {
        printf("DOUBLE free addr: %p\n", ptr);
        return;
    }

    // ======== terminal-console way ========
    printf("[-%s:%d], addr: %p\n", file, line, ptr);
    free(ptr);
}

// 请深入理解GCC的编译流程, 理解 why #define 只能放在这里并同时生效
#    define malloc(size) _malloc(size, __FILE__, __LINE__)
#    define free(ptr)    _free(ptr, __FILE__, __LINE__)
// calloc,realloc, new/delete

#else

typedef void *(*malloc_t)(size_t size);
malloc_t malloc_func = NULL;

typedef void (*free_t)(void *ptr);
free_t free_func = NULL;

// hook and dlsym(linux 系统函数)
/* dlsym 用于在动态链接库中查找符号(函数或变量).
 * RTLD_NEXT 是一个特殊的句柄,表示在共享库查找链中查找下一个共享库.
 * 通过使用 RTLD_NEXT, 可以实现函数拦截,同时保留对原始实现的访问.
 * 这种技术在调试、性能监控、日志记录等场景中非常有用，
 * 通过拦截函数调用,可以在不修改原始代码的情况下添加额外的功能.
 * 
 ? Windows Hook技术和Linux实现有点差异, 本质原理想通.
 * 
 */
bool enable_malloc_hook = true;
bool enable_free_hook   = true;

void *malloc(size_t size)
{
    if (enable_malloc_hook)
    {
        enable_malloc_hook = false; /* sprintf 内部实现有用到内存申请 malloc,避免递归调用 */

        void *ptr = malloc_func(size);

        // 函数调用堆栈指针, 通过命令解析出对应的源代码哪一行调用
        // addr2line -f -e MemoryLeak.out -a caller[0x400754]
        void *caller = __builtin_return_address(0);

        // ======== log-file way ========
        char buff[128] = {0};
        sprintf(buff, "./mem/%p.mem", ptr);

        FILE *fp = fopen(buff, "w");
        // fprintf(fp, "[+%s:%d] ---> addr:%lld\n", __FILE__, __LINE__, size);
        // fprintf(fp, "[+%p] ---> addr:%p, size:%lld\n", caller, ptr, size);
        fprintf(fp, "[+%p] ---> addr:%p, size:%ld\n", caller, ptr, size);
        fflush(fp);
        fclose(fp);

        // ======== terminal-console way ========
        // printf("[+%s:%d], memory-bytes size: %ld\n", __FILE__, __LINE__, size);
        // printf("[+%p] ---> addr:%lld\n", caller, size);
        printf("[+%p] ---> addr:%p\n", caller, ptr);

        enable_malloc_hook = true;
        return ptr;
    }
    else
    {
        return malloc_func(size);
    }
}

void free(void *ptr)
{
    if (enable_free_hook)
    {
        enable_free_hook = false;
        void *caller     = __builtin_return_address(0);

        // ======== log-file way ========
        char buff[128] = {0};
        sprintf(buff, "./mem/%p.mem", ptr);

        if (-1 == unlink(buff)) /* double free */
        {
            printf("DOUBLE free addr: %p\n", ptr);
            return;
        }

        // ======== terminal-console way ========
        // printf("[-%s:%d], addr: %p\n", __FILE__, line, ptr);
        printf("[-%p], addr: %p\n", caller, ptr);

        free_func(ptr);

        enable_free_hook = true;
    }
    else
    {
        free_func(ptr);
    }
}

void init_hook(void)
{
    if (NULL == malloc_func)
        // malloc_func = dlsym(RTLD_NEXT, "malloc");
        malloc_func = (malloc_t)dlsym(RTLD_NEXT, "malloc");

    if (NULL == free_func)
        // free_func = dlsym(RTLD_NEXT, "free");
        free_func = (free_t)dlsym(RTLD_NEXT, "free");
}

#    define DEBUG_MEMROY_LEAK init_hook();

#endif /* 单文件检测 */

// ------------------------------------
int main(int argc, const char **argv)
{
    // ===================================================
    // void *ptr1 = malloc(16);
    // if (NULL == ptr1)
    //     std::cout << "malloc is NOT successfully\n";

    // void *ptr2 = malloc(32);
    // if (NULL == ptr2)
    //     std::cout << "malloc is NOT successfully\n";

    // if (NULL != ptr1)
    //     free(ptr1);
    // ===================================================

    // =====================
    // gcc MemoryLeak.cpp -ldl
    // DEBUG_MEMROY_LEAK // 整个应用程序的内存申请和释放都被hook
    // =====================

    void *ptr1 = malloc(16);
    if (NULL == ptr1)
        // std::cout << "malloc is NOT successfully\n";
        printf("malloc is NOT successfully\n");

    void *ptr2 = malloc(32);
    if (NULL == ptr2)
        // std::cout << "malloc is NOT successfully\n";
        printf("malloc is NOT successfully\n");

    if (NULL != ptr1)
        free(ptr1);

    return 0;
}
