/**
 * @file 16_interface_filter.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 抛砖引玉: 接口设计与回调函数
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o filter_example.exe 16_interface_filter.c
 * clang -o filter_example.exe 16_interface_filter.c
 * 
 */

#include <Windows.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义回调函数类型：判断元素是否保留（玉）
typedef bool (*Predicate)(const void *element, void *context);

// 通用过滤器（抛出的砖）：遍历数组，将满足条件的元素复制到新数组
void *filter(const void *base, size_t count, size_t elem_size, Predicate pred, void *context, size_t *out_count)
{
    if (!base || !pred || !out_count)
        return NULL;

    // 第一遍：统计满足条件的元素个数
    size_t good_count = 0;
    const char *byte_base = (const char *)base;
    for (size_t i = 0; i < count; ++i)
    {
        const void *elem = byte_base + i * elem_size;
        if (pred(elem, context))
            good_count++;
    }

    // 分配结果数组
    void *result = malloc(good_count * elem_size);
    if (!result)
    {
        *out_count = 0;
        return NULL;
    }

    // 第二遍：复制满足条件的元素
    char *out_ptr = (char *)result;
    for (size_t i = 0; i < count; ++i)
    {
        const void *elem = byte_base + i * elem_size;
        if (pred(elem, context))
        {
            memcpy(out_ptr, elem, elem_size);
            out_ptr += elem_size;
        }
    }
    *out_count = good_count;
    return result;
}

// ========== 用户自定义的“玉” ==========
// 示例1：判断整数是否为正数
bool is_positive(const void *elem, void *ctx)
{
    (void)ctx; // 未使用上下文
    int value = *(const int *)elem;
    return value > 0;
}

// 示例2：判断字符串长度是否大于阈值（使用上下文传递阈值）
bool string_length_gt(const void *elem, void *ctx)
{
    size_t threshold = *(size_t *)ctx;
    const char *str = *(const char **)elem; // 数组元素是 char*
    return strlen(str) > threshold;
}

// 示例3：判断整数是否为偶数（另一个玉）
bool is_even(const void *elem, void *ctx)
{
    (void)ctx;
    return (*(const int *)elem) % 2 == 0;
}

// 打印整数数组
void print_int_array(const int *arr, size_t n)
{
    printf("[");
    for (size_t i = 0; i < n; ++i)
    {
        printf("%d%s", arr[i], i + 1 < n ? ", " : "");
    }
    printf("]\n");
}

// 打印字符串数组
void print_str_array(const char **arr, size_t n)
{
    printf("[");
    for (size_t i = 0; i < n; ++i)
    {
        printf("\"%s\"%s", arr[i], i + 1 < n ? ", " : "");
    }
    printf("]\n");
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 示例1：过滤整数数组中的正数
    int numbers[] = {-3, 5, -1, 0, 8, -2, 7};
    size_t count = sizeof(numbers) / sizeof(numbers[0]);
    size_t filtered_count;

    int *positives = (int *)filter(numbers, count, sizeof(int), is_positive, NULL, &filtered_count);
    if (positives)
    {
        printf("原数组: ");
        print_int_array(numbers, count);
        printf("正数过滤结果: ");
        print_int_array(positives, filtered_count);
        free(positives);
    }

    // 示例2：过滤字符串数组，保留长度大于3的字符串
    const char *words[] = {"a", "hello", "cat", "elephant", "hi", "bird"};
    count = sizeof(words) / sizeof(words[0]);
    size_t threshold = 3;

    const char **long_words =
        (const char **)filter(words, count, sizeof(const char *), string_length_gt, &threshold, &filtered_count);
    if (long_words)
    {
        printf("\n原字符串数组: ");
        print_str_array(words, count);
        printf("长度 > %zu 的单词: ", threshold);
        print_str_array(long_words, filtered_count);
        free(long_words);
    }

    // 示例3：使用同一个 filter 框架，传入不同的“玉”——偶数筛选
    int *evens = (int *)filter(numbers, count, sizeof(int), is_even, NULL, &filtered_count);
    if (evens)
    {
        printf("\n偶数过滤结果: ");
        print_int_array(evens, filtered_count);
        free(evens);
    }

// 进阶：使用 _Generic 包装，让接口更友好（抛砖引玉的语法糖）
#define FILTER_ARRAY(arr, pred, ctx)                                                                                   \
    _Generic((arr),                                                                                                    \
        int *: filter((arr), sizeof(arr) / sizeof(*(arr)), sizeof(*(arr)), pred, ctx, &(size_t){0}),                   \
        const char **: filter((arr), sizeof(arr) / sizeof(*(arr)), sizeof(*(arr)), pred, ctx, &(size_t){0}))
    // 注意：上面简化了，实际需返回计数，此处仅示意

    printf("\n抛砖引玉核心思想：\n");
    printf("- 框架提供通用算法（砖），用户提供回调函数（玉）\n");
    printf("- 通过函数指针解耦，同一接口适配无数种行为\n");
    printf("- 上下文参数 void* 允许传递自定义状态，增强灵活性\n");

    return 0;
}
