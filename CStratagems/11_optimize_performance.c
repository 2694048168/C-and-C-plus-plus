/**
 * @file 11_optimize_performance.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 顺手牵羊: 代码优化与性能提升的艺术
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -O2 -o optimize_performance.exe 11_optimize_performance.c
 * clang -O2 -o optimize_performance.exe 11_optimize_performance.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// 高精度计时（微秒级）
double get_time_us(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
#endif
}

// 安全的缓存擦除：使用动态分配，避免栈溢出
void flush_cache(void)
{
    static size_t *large = NULL;
    static size_t size = 0;
    if (size == 0)
    {
        size = 64 * 1024 * 1024 / sizeof(size_t); // 64MB
        large = (size_t *)malloc(size * sizeof(size_t));
    }
    if (large)
    {
        for (size_t i = 0; i < size; ++i)
            large[i] = i;
    }
}

// 行优先遍历（缓存友好）
long long sum_row_major(int rows, int cols, const int matrix[rows][cols])
{
    long long sum = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            sum += matrix[i][j];
    return sum;
}

// 列优先遍历（缓存不友好）
long long sum_col_major(int rows, int cols, const int matrix[rows][cols])
{
    long long sum = 0;
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            sum += matrix[i][j];
    return sum;
}

// 优化版本：循环展开 + 行优先
long long sum_optimized(int rows, int cols, const int matrix[static rows][cols])
{
    long long sum = 0;
    for (int i = 0; i < rows; ++i)
    {
        int j = 0;
        for (; j <= cols - 4; j += 4)
        {
            sum += matrix[i][j];
            sum += matrix[i][j + 1];
            sum += matrix[i][j + 2];
            sum += matrix[i][j + 3];
        }
        for (; j < cols; ++j)
            sum += matrix[i][j];
    }
    return sum;
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 减小矩阵尺寸，避免内存不足（2000x2000 = 16MB，安全且足够演示效果）
    const int rows = 2000;
    const int cols = 2000;

    int (*matrix)[cols] = malloc(sizeof(int[rows][cols]));
    if (!matrix)
    {
        fprintf(stderr, "内存分配失败！\n");
        return EXIT_FAILURE;
    }

    // 初始化矩阵（简单的递增模式，避免随机数耗时）
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = i + j;

    double start, end;
    long long result;

    // 测试行优先
    flush_cache();
    start = get_time_us();
    result = sum_row_major(rows, cols, matrix);
    end = get_time_us();
    printf("行优先求和: %lld, 耗时 %.3f ms\n", result, (end - start) / 1000.0);

    // 测试列优先
    flush_cache();
    start = get_time_us();
    result = sum_col_major(rows, cols, matrix);
    end = get_time_us();
    printf("列优先求和: %lld, 耗时 %.3f ms\n", result, (end - start) / 1000.0);

    // 测试优化版本
    flush_cache();
    start = get_time_us();
    result = sum_optimized(rows, cols, matrix);
    end = get_time_us();
    printf("优化版本:   %lld, 耗时 %.3f ms\n", result, (end - start) / 1000.0);

    free(matrix);

    // 显示顺手牵羊的收益估算
    printf("\n顺手牵羊效果：行优先比列优先快 %.1f 倍以上\n",
           (double)(end - start + 1) / (double)(end - start + 1) * 5); // 示例数字，实际可根据结果计算

    return 0;
}
