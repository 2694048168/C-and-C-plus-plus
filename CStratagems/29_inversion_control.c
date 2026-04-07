/**
 * @file 29_inversion_control.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 反客为主: 用户通过回调接口控制库执行流程
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o inversion_control.exe 29_inversion_control.c
 * clang -o inversion_control.exe 29_inversion_control.c
 *
 */

#include <Windows.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== 库接口（提供骨架，不控制具体逻辑） ==========
// 用户回调类型：对每个元素进行处理，返回是否保留，并可修改值
typedef bool (*TransformFunc)(void *ctx, int *value);

// 库函数：遍历数组，应用用户回调，生成新数组（反客为主的核心）
int *filter_map(const int *src, size_t n, TransformFunc transform, void *ctx, size_t *out_n)
{
    if (!src || !transform || !out_n)
        return NULL;

    // 第一遍：确定输出数组大小
    size_t count = 0;
    for (size_t i = 0; i < n; ++i)
    {
        int val = src[i];
        if (transform(ctx, &val))
        {
            count++;
        }
    }

    // 分配结果数组
    int *result = (int *)malloc(count * sizeof(int));
    if (!result)
    {
        *out_n = 0;
        return NULL;
    }

    // 第二遍：填充结果
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i)
    {
        int val = src[i];
        if (transform(ctx, &val))
        {
            result[idx++] = val;
        }
    }
    *out_n = count;
    return result;
}

// ========== 用户代码（反客为主：用户决定过滤和转换规则） ==========
// 用户上下文：可以携带额外参数
typedef struct
{
    int threshold;
    int multiplier;
} UserContext;

// 用户回调：只保留大于阈值的元素，并将其乘以乘数
bool my_filter_transform(void *ctx, int *value)
{
    UserContext *uc = (UserContext *)ctx;
    if (*value > uc->threshold)
    {
        *value *= uc->multiplier;
        return true;
    }
    return false;
}

// 另一个用户回调：只保留偶数，并加上固定值
bool even_plus(void *ctx, int *value)
{
    (void)ctx; // 未使用上下文
    if (*value % 2 == 0)
    {
        *value += 100;
        return true;
    }
    return false;
}

// 主程序：演示反客为主
int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    int data[] = {1, 5, 2, 8, 3, 10, 4};
    size_t n = sizeof(data) / sizeof(data[0]);

    printf("原始数组: ");
    for (size_t i = 0; i < n; ++i)
        printf("%d ", data[i]);
    printf("\n");

    // 反客为主1：用户定义自己的处理规则（过滤大于3并乘以2）
    UserContext ctx1 = {.threshold = 3, .multiplier = 2};
    size_t out_n1;
    int *result1 = filter_map(data, n, my_filter_transform, &ctx1, &out_n1);
    if (result1)
    {
        printf("大于3且乘以2: ");
        for (size_t i = 0; i < out_n1; ++i)
            printf("%d ", result1[i]);
        printf("\n");
        free(result1);
    }

    // 反客为主2：另一个规则（保留偶数并加100）
    size_t out_n2;
    int *result2 = filter_map(data, n, even_plus, NULL, &out_n2);
    if (result2)
    {
        printf("偶数加100: ");
        for (size_t i = 0; i < out_n2; ++i)
            printf("%d ", result2[i]);
        printf("\n");
        free(result2);
    }

    printf("\n反客为主核心：\n");
    printf("- 库提供通用 filter_map 骨架\n");
    printf("- 用户通过回调注入具体逻辑（反客为主）\n");
    printf("- 库不依赖用户，用户控制库的行为\n");
    printf("- 上下文指针 void* 允许传递任意状态\n");

    return 0;
}
