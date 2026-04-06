/**
 * @file 13_reuse_refactoring.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 借尸还魂: 资源复用与代码重构
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o reuse_refactoring.exe 13_reuse_refactoring.c
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// 代表可复用的“尸体”结构
typedef struct
{
    int id;
    char name[32];
    bool active; // 是否正在使用（还魂标志）
} Resource;

// 对象池
typedef struct
{
    Resource *pool; // 资源数组（尸体存放地）
    int capacity;   // 池容量
    int used;       // 当前活跃对象数（还魂数量）
} ResourcePool;

// 创建池：预先分配一堆“尸体”
ResourcePool *pool_create(int capacity)
{
    ResourcePool *rp = (ResourcePool *)malloc(sizeof(ResourcePool));
    if (!rp)
        return NULL;
    rp->pool = (Resource *)calloc(capacity, sizeof(Resource));
    if (!rp->pool)
    {
        free(rp);
        return NULL;
    }
    rp->capacity = capacity;
    rp->used = 0;
    // 初始时所有资源未激活（尸体状态）
    for (int i = 0; i < capacity; ++i)
    {
        rp->pool[i].active = false;
        rp->pool[i].id = -1;
    }
    return rp;
}

// 从池中“借尸还魂”：取出一个未使用的资源，赋予新生命
Resource *pool_acquire(ResourcePool *rp, int id, const char *name)
{
    if (!rp || rp->used >= rp->capacity)
        return NULL;
    // 查找第一个未激活的“尸体”
    for (int i = 0; i < rp->capacity; ++i)
    {
        if (!rp->pool[i].active)
        {
            Resource *res = &rp->pool[i];
            res->active = true;
            res->id = id;
            strncpy(res->name, name, sizeof(res->name) - 1);
            res->name[sizeof(res->name) - 1] = '\0';
            rp->used++;
            printf("[借尸还魂] 资源 #%d 被激活为 (id=%d, name=%s)\n", i, id, name);
            return res;
        }
    }
    return NULL;
}

// 归还资源（重新变为尸体）
void pool_release(ResourcePool *rp, Resource *res)
{
    if (!rp || !res || !res->active)
        return;
    res->active = false;
    rp->used--;
    printf("[归于尸体] 资源 (id=%d, name=%s) 已释放，可再次借用\n", res->id, res->name);
    // 可选：清空数据，防止残留信息
    res->id = -1;
    res->name[0] = '\0';
}

// 销毁池（将所有尸体超度）
void pool_destroy(ResourcePool *rp)
{
    if (rp)
    {
        free(rp->pool);
        free(rp);
    }
}

// 模拟使用资源
void use_resource(Resource *res)
{
    if (res && res->active)
    {
        printf("  使用资源: id=%d, name=%s\n", res->id, res->name);
    }
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 创建一个容量为3的对象池（3具尸体）
    ResourcePool *pool = pool_create(3);
    if (!pool)
        return EXIT_FAILURE;

    // 借尸还魂：激活资源
    Resource *r1 = pool_acquire(pool, 101, "Alice");
    Resource *r2 = pool_acquire(pool, 102, "Bob");
    Resource *r3 = pool_acquire(pool, 103, "Charlie");
    Resource *r4 = pool_acquire(pool, 104, "David"); // 池已满，返回NULL
    if (!r4)
        printf("[失败] 池已满，无法借尸\n");

    use_resource(r1);
    use_resource(r2);

    // 归还一个资源（变回尸体）
    pool_release(pool, r2);

    // 现在可以再次借尸：之前 Bob 的尸体被重新激活为新的资源
    Resource *r5 = pool_acquire(pool, 201, "Eve");
    use_resource(r5);

    // 最终销毁池
    pool_destroy(pool);

    // 额外演示：代码重构中的“借尸还魂” —— 用函数指针复用旧逻辑
    printf("\n===== 重构示例：函数指针复用 =====\n");
    // 旧的排序函数（只能升序）
    void old_sort(int *arr, int n)
    {
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                if (arr[i] > arr[j])
                {
                    int tmp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = tmp;
                }
        printf("旧排序完成（升序）\n");
    }
    // 新需求：需要降序或自定义比较。借旧函数的尸，还魂为通用排序
    typedef int (*CompareFunc)(int, int);
    int cmp_desc(int a, int b)
    {
        return b - a;
    }
    void generic_sort(int *arr, int n, CompareFunc cmp)
    {
        // 复用旧排序的框架，只修改比较逻辑
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                if (cmp(arr[i], arr[j]) > 0)
                {
                    int tmp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = tmp;
                }
        printf("通用排序完成\n");
    }
    int data[] = {3, 1, 4, 1, 5, 9};
    generic_sort(data, 6, cmp_desc);
    for (int i = 0; i < 6; ++i)
        printf("%d ", data[i]);
    printf("\n");

    return 0;
}
