/**
 * @file 06_malloc_free.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 无中生有: 动态内存分配和管理的艺术
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 06_malloc_free.c -o 06_malloc_free.exe
 * clang 06_malloc_free.c -o 06_malloc_free.exe
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// 动态整数数组结构（不透明设计，仅内部可见）
typedef struct
{
    int *data;       // 堆上实际存储区
    size_t size;     // 当前元素个数
    size_t capacity; // 当前已分配容量
} IntVector;

// 创建空向量（无中生有）
IntVector *vec_create(void)
{
    IntVector *v = (IntVector *)malloc(sizeof(IntVector));
    if (!v)
        return NULL; // 分配失败
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
    return v;
}

// 销毁向量并释放所有内存（有归于无）
void vec_destroy(IntVector *v)
{
    if (v)
    {
        free(v->data); // 释放元素存储区
        free(v);       // 释放结构体本身
    }
}

// 确保容量至少为 new_cap（内部扩容逻辑）
static bool vec_reserve(IntVector *v, size_t new_cap)
{
    if (new_cap <= v->capacity)
        return true;

    size_t target = v->capacity;
    if (target == 0)
        target = 4; // 初始容量
    while (target < new_cap)
        target *= 2; // 翻倍策略

    int *new_data = (int *)realloc(v->data, target * sizeof(int));
    if (!new_data)
        return false;

    v->data = new_data;
    v->capacity = target;
    return true;
}

// 在末尾添加元素（动态扩容）
bool vec_push_back(IntVector *v, int value)
{
    if (!v)
        return false;
    if (!vec_reserve(v, v->size + 1))
        return false;
    v->data[v->size++] = value;
    return true;
}

// 获取元素（带边界检查）
bool vec_get(const IntVector *v, size_t index, int *out)
{
    if (!v || !out || index >= v->size)
        return false;
    *out = v->data[index];
    return true;
}

// 返回当前大小
size_t vec_size(const IntVector *v)
{
    return v ? v->size : 0;
}

// 遍历并执行回调（演示无中生有后的使用）
void vec_foreach(const IntVector *v, void (*func)(int))
{
    if (!v || !func)
        return;
    for (size_t i = 0; i < v->size; ++i)
    {
        func(v->data[i]);
    }
}

// 示例：打印元素
void print_int(int x)
{
    printf("%d ", x);
}

int main(int argc, char const *argv[])
{
    SetConsoleOutputCP(CP_UTF8); // 设置输出代码页为 UTF-8

    // 无中生有：创建空向量
    IntVector *vec = vec_create();
    if (!vec)
    {
        fprintf(stderr, "创建向量失败\n");
        return EXIT_FAILURE;
    }

    // 动态添加元素（从无到有，逐步增长）
    for (int i = 1; i <= 10; ++i)
    {
        if (!vec_push_back(vec, i * i))
        {
            fprintf(stderr, "添加元素 %d 失败\n", i * i);
            vec_destroy(vec);
            return EXIT_FAILURE;
        }
        printf("已添加 %d, 当前大小=%zu, 容量=%zu\n", i * i, vec_size(vec), vec->capacity);
    }

    // 随机访问
    int val;
    if (vec_get(vec, 5, &val))
    {
        printf("\n索引5处的值为: %d\n", val);
    }

    // 遍历所有元素
    printf("向量所有元素: ");
    vec_foreach(vec, print_int);
    printf("\n");

    // 有归于无：彻底销毁
    vec_destroy(vec);
    vec = NULL; // 防止悬挂指针

    return EXIT_SUCCESS;
}
