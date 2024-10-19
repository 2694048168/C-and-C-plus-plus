/**
 * @file ASan_memory.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 要使用ASan, 需要使用支持ASan的编译器, 如Clang或GCC, 并开启ASan相关的编译选项
 * clang -fsanitize=address -g ASan_memory.c -o ASan_memory
 * 
 * gcc -fsanitize=address -g ASan_memory.c -o ASan_memory
 * 
 * ASan-sanitizers 其他选项
 * 除了 -fsanitize=address 外, 还有其他 AddressSanitizer 相关的编译选项可供选择.
 * 1. Memory Sanitizer (-fsanitize=memory):
 *   用于检测对未初始化内存或使用已释放内存的操作, 这个选项可以帮助发现一些难以察觉的内存错误.
 * 2. UndefinedBehaviorSanitizer (-fsanitize=undefined):
 *   用于检测未定义行为, 例如整数溢出、空指针解引用等问题, 这有助于发现代码中的潜在 bug.
 * 3. Thread Sanitizer (-fsanitize=thread):
 *   用于检测多线程程序中的数据竞争和死锁问题, 这个选项可以帮助识别并修复多线程程序中的并发 bug.
 * 4. Address Sanitizer with Leak Detection (-fsanitize=leak):
 *   启用 AddressSanitizer 的同时, 也检测内存泄漏问题, 这个选项有助于发现代码中的内存泄漏 bug.
 * 5. Coverage Sanitizer (-fsanitize=coverage):
 *   用于生成代码覆盖率报告, 检测程序中哪些部分被执行过, 这个选项通常用于代码覆盖率测试和分析.
 * 6. Kernel Address Sanitizer (-fsanitize=kernel-address):
 *   针对 Linux 内核模块开发, 用于检测内核中的内存错误.
 * 
 * MSVC: https://learn.microsoft.com/zh-cn/cpp/sanitizers/asan?view=msvc-170
 * 
 */

void leak_memory()
{
    int *p = (int *)malloc(sizeof(int) * 10);
    // 没有释放内存，导致内存泄漏
}

int main(int argc, const char **argv)
{
    int *arr = (int *)malloc(sizeof(int) * 5);
    arr[5]   = 10; // 内存访问越界错误

    free(arr); // 使用 free 释放内存

    int *p = (int *)malloc(sizeof(int));
    free(p); // 使用 free 释放内存

    int *q = NULL;
    *q     = 5; // 使用空指针访问内存错误

    leak_memory();

    return 0;
}
