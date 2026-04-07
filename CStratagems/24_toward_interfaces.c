/**
 * @file 24_toward_interfaces.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 偷梁换柱: 数据结构的替换与抽象策略(面向接口编程)
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 24_toward_interfaces.c -o 24_toward_interfaces.exe
 * clang 24_toward_interfaces.c -o 24_toward_interfaces.exe
 *
 */

#include <Windows.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== 集合接口（偷梁换柱的统一契约） ==========
// 前向声明
typedef struct Set Set;

// 集合操作虚表
typedef struct
{
    bool (*add)(Set *self, const char *elem);
    bool (*contains)(const Set *self, const char *elem);
    void (*remove)(Set *self, const char *elem);
    size_t (*size)(const Set *self);
    void (*destroy)(Set *self);
} SetVTable;

// 集合对象结构（包含虚表指针）
struct Set
{
    SetVTable *vtable;
};

// 接口宏（调用虚函数）
#define SET_ADD(s, e) ((s)->vtable->add((s), (e)))
#define SET_CONTAINS(s, e) ((s)->vtable->contains((s), (e)))
#define SET_REMOVE(s, e) ((s)->vtable->remove((s), (e)))
#define SET_SIZE(s) ((s)->vtable->size((s)))
#define SET_DESTROY(s) ((s)->vtable->destroy((s)))

// ========== 实现1：基于哈希表的集合（一种梁柱） ==========
typedef struct HashNode
{
    char *key;
    struct HashNode *next;
} HashNode;

typedef struct
{
    Set base;
    HashNode **buckets;
    size_t bucket_count;
    size_t count;
} HashSet;

// 哈希函数（简单示例）
static unsigned hash(const char *str)
{
    unsigned h = 0;
    while (*str)
        h = h * 31 + (unsigned)(*str++);
    return h;
}

static bool hashset_add(Set *self, const char *elem)
{
    HashSet *hs = (HashSet *)self;
    unsigned h = hash(elem) % hs->bucket_count;
    // 检查是否已存在
    for (HashNode *node = hs->buckets[h]; node; node = node->next)
    {
        if (strcmp(node->key, elem) == 0)
            return false;
    }
    // 插入新节点
    HashNode *node = malloc(sizeof(HashNode));
    node->key = strdup(elem);
    node->next = hs->buckets[h];
    hs->buckets[h] = node;
    hs->count++;
    return true;
}

static bool hashset_contains(const Set *self, const char *elem)
{
    const HashSet *hs = (const HashSet *)self;
    unsigned h = hash(elem) % hs->bucket_count;
    for (HashNode *node = hs->buckets[h]; node; node = node->next)
    {
        if (strcmp(node->key, elem) == 0)
            return true;
    }
    return false;
}

static void hashset_remove(Set *self, const char *elem)
{
    HashSet *hs = (HashSet *)self;
    unsigned h = hash(elem) % hs->bucket_count;
    HashNode *prev = NULL;
    for (HashNode *node = hs->buckets[h]; node; node = node->next)
    {
        if (strcmp(node->key, elem) == 0)
        {
            if (prev)
                prev->next = node->next;
            else
                hs->buckets[h] = node->next;
            free(node->key);
            free(node);
            hs->count--;
            return;
        }
        prev = node;
    }
}

static size_t hashset_size(const Set *self)
{
    const HashSet *hs = (const HashSet *)self;
    return hs->count;
}

static void hashset_destroy(Set *self)
{
    HashSet *hs = (HashSet *)self;
    for (size_t i = 0; i < hs->bucket_count; ++i)
    {
        HashNode *node = hs->buckets[i];
        while (node)
        {
            HashNode *next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
    }
    free(hs->buckets);
    free(hs);
}

static SetVTable hashset_vtable = {.add = hashset_add,
                                   .contains = hashset_contains,
                                   .remove = hashset_remove,
                                   .size = hashset_size,
                                   .destroy = hashset_destroy};

Set *create_hashset(size_t bucket_count)
{
    HashSet *hs = malloc(sizeof(HashSet));
    hs->base.vtable = &hashset_vtable;
    hs->buckets = calloc(bucket_count, sizeof(HashNode *));
    hs->bucket_count = bucket_count;
    hs->count = 0;
    return (Set *)hs;
}

// ========== 实现2：基于有序动态数组的集合（另一种梁柱） ==========
typedef struct
{
    Set base;
    char **array;
    size_t size;
    size_t capacity;
} ArraySet;

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char **)a, *(const char **)b);
}

static bool array_add(Set *self, const char *elem)
{
    ArraySet *as = (ArraySet *)self;
    // 检查是否存在（二分查找）
    char **found = bsearch(&elem, as->array, as->size, sizeof(char *), cmp_str);
    if (found)
        return false;
    // 扩容
    if (as->size >= as->capacity)
    {
        size_t new_cap = as->capacity ? as->capacity * 2 : 4;
        char **new_arr = realloc(as->array, new_cap * sizeof(char *));
        if (!new_arr)
            return false;
        as->array = new_arr;
        as->capacity = new_cap;
    }
    // 插入并保持有序
    size_t pos = 0;
    while (pos < as->size && strcmp(as->array[pos], elem) < 0)
        pos++;
    memmove(&as->array[pos + 1], &as->array[pos], (as->size - pos) * sizeof(char *));
    as->array[pos] = strdup(elem);
    as->size++;
    return true;
}

static bool array_contains(const Set *self, const char *elem)
{
    const ArraySet *as = (const ArraySet *)self;
    return bsearch(&elem, as->array, as->size, sizeof(char *), cmp_str) != NULL;
}

static void array_remove(Set *self, const char *elem)
{
    ArraySet *as = (ArraySet *)self;
    char **found = bsearch(&elem, as->array, as->size, sizeof(char *), cmp_str);
    if (!found)
        return;
    size_t idx = found - as->array;
    free(as->array[idx]);
    memmove(&as->array[idx], &as->array[idx + 1], (as->size - idx - 1) * sizeof(char *));
    as->size--;
}

static size_t array_size(const Set *self)
{
    const ArraySet *as = (const ArraySet *)self;
    return as->size;
}

static void array_destroy(Set *self)
{
    ArraySet *as = (ArraySet *)self;
    for (size_t i = 0; i < as->size; ++i)
        free(as->array[i]);
    free(as->array);
    free(as);
}

static SetVTable array_vtable = {
    .add = array_add, .contains = array_contains, .remove = array_remove, .size = array_size, .destroy = array_destroy};

Set *create_arrayset(void)
{
    ArraySet *as = malloc(sizeof(ArraySet));
    as->base.vtable = &array_vtable;
    as->array = NULL;
    as->size = 0;
    as->capacity = 0;
    return (Set *)as;
}

// ========== 上层调用代码（完全不依赖底层实现） ==========
void demo_set(Set *set, const char *name)
{
    printf("\n=== 使用 %s ===\n", name);
    SET_ADD(set, "apple");
    SET_ADD(set, "banana");
    SET_ADD(set, "cherry");
    SET_ADD(set, "apple"); // 重复添加应失败
    printf("是否包含 'banana'? %s\n", SET_CONTAINS(set, "banana") ? "是" : "否");
    printf("是否包含 'grape'? %s\n", SET_CONTAINS(set, "grape") ? "是" : "否");
    printf("集合大小: %zu\n", SET_SIZE(set));
    SET_REMOVE(set, "banana");
    printf("删除 'banana' 后大小: %zu\n", SET_SIZE(set));
}

int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 偷梁换柱：创建两种底层实现，但上层接口完全相同
    Set *hash_set = create_hashset(16);
    Set *array_set = create_arrayset();

    demo_set(hash_set, "哈希表集合");
    demo_set(array_set, "有序数组集合");

    SET_DESTROY(hash_set);
    SET_DESTROY(array_set);

    printf("\n偷梁换柱核心思想：\n");
    printf("- 通过虚表（函数指针）定义统一接口\n");
    printf("- 上层代码只依赖接口，不依赖具体数据结构\n");
    printf("- 可以随时更换底层实现（哈希表 ↔ 有序数组）\n");
    printf("- 符合开闭原则：对扩展开放，对修改关闭\n");
    return 0;
}
