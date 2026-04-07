/**
 * @file 21_vec.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 21_vec.exe 21_vec.c
 * clang -o 21_vec.exe 21_vec.c
 *
 */

#include "21_vec.h"
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 真正的结构定义（藏在 .c 中，用户不可见）
struct Vector
{
    int *data;
    size_t size;
    size_t capacity;
};

// 静态辅助函数（内聚，不对外暴露）
static bool vec_reserve(Vector *v, size_t new_cap)
{
    if (new_cap <= v->capacity)
        return true;
    size_t target = v->capacity;
    if (target == 0)
        target = 4;
    while (target < new_cap)
        target *= 2;
    int *new_data = (int *)realloc(v->data, target * sizeof(int));
    if (!new_data)
        return false;
    v->data = new_data;
    v->capacity = target;
    return true;
}

// 公开接口实现
Vector *vec_create(void)
{
    Vector *v = (Vector *)malloc(sizeof(Vector));
    if (!v)
        return NULL;
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
    return v;
}

void vec_destroy(Vector *v)
{
    if (v)
    {
        free(v->data);
        free(v);
    }
}

bool vec_push_back(Vector *v, int value)
{
    if (!v)
        return false;
    if (!vec_reserve(v, v->size + 1))
        return false;
    v->data[v->size++] = value;
    return true;
}

bool vec_pop_back(Vector *v, int *out)
{
    if (!v || v->size == 0)
        return false;
    if (out)
        *out = v->data[--v->size];
    else
        v->size--;
    return true;
}

bool vec_get(const Vector *v, size_t index, int *out)
{
    if (!v || !out || index >= v->size)
        return false;
    *out = v->data[index];
    return true;
}

size_t vec_size(const Vector *v)
{
    return v ? v->size : 0;
}

bool vec_is_empty(const Vector *v)
{
    return v ? v->size == 0 : true;
}

void vec_foreach(const Vector *v, VecVisitFunc func, void *ctx)
{
    if (!v || !func)
        return;
    for (size_t i = 0; i < v->size; ++i)
    {
        func(v->data[i], ctx);
    }
}

// -------------------------------
// 自定义遍历打印函数
void print_int(int value, void *ctx)
{
    printf("%d ", value);
}

// 求和函数
void sum_int(int value, void *ctx)
{
    int *total = (int *)ctx;
    *total += value;
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 创建向量（只知道是 Vector*，不知道内部结构）
    Vector *v = vec_create();
    if (!v)
        return 1;

    // 添加元素
    for (int i = 1; i <= 10; ++i)
    {
        vec_push_back(v, i * i);
    }

    printf("向量元素: ");
    vec_foreach(v, print_int, NULL);
    printf("\n");

    int total = 0;
    vec_foreach(v, sum_int, &total);
    printf("元素和: %d\n", total);

    // 弹出最后一个元素
    int last;
    if (vec_pop_back(v, &last))
    {
        printf("弹出元素: %d, 剩余大小: %zu\n", last, vec_size(v));
    }

    // 随机访问
    int val;
    if (vec_get(v, 2, &val))
    {
        printf("索引2的值: %d\n", val);
    }

    vec_destroy(v);
    return 0;
}
