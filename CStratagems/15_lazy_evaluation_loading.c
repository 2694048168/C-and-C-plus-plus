/**
 * @file 15_lazy_evaluation_loading.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 欲擒故纵: 延迟计算和懒加载技巧
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 15_lazy_evaluation_loading.c -o 15_lazy_evaluation_loading.exe
 * clang 15_lazy_evaluation_loading.c -o 15_lazy_evaluation_loading.exe
 * 
 */

#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ========== 示例1：懒加载单例（欲擒故纵：先声明，使用时才创建） ==========
typedef struct
{
    int id;
    char data[256];
} ExpensiveObject;

static ExpensiveObject *lazy_object = NULL;

ExpensiveObject *get_object(void)
{
    if (!lazy_object)
    { // 纵：第一次调用才分配
        printf("[懒加载] 正在创建昂贵对象...\n");
        lazy_object = (ExpensiveObject *)malloc(sizeof(ExpensiveObject));
        lazy_object->id = rand() % 1000;
        snprintf(lazy_object->data, sizeof(lazy_object->data), "初始数据 %d", lazy_object->id);
    }
    else
    {
        printf("[懒加载] 对象已存在，直接返回\n");
    }
    return lazy_object;
}

void destroy_object(void)
{
    if (lazy_object)
    {
        free(lazy_object);
        lazy_object = NULL;
        printf("[清理] 对象已销毁\n");
    }
}

// ========== 示例2：惰性求值宏（参数仅当需要时才计算） ==========
// 传统宏：参数总会计算，有副作用
#define SQUARE_EAGER(x) ((x) * (x))

// 惰性宏：借助 _Generic 或 临时变量实现一次求值（C11）
#define LAZY_SQUARE(x)                                                                                                 \
    ({                                                                                                                 \
        __typeof__(x) _tmp = (x);                                                                                      \
        _tmp *_tmp;                                                                                                    \
    })

// 更通用的惰性求值：使用静态内联函数（推荐）
static inline double lazy_pow(double base, int exp)
{
    // 模拟昂贵计算
    printf("   [惰性计算] 正在计算 %.2f^%d ...\n", base, exp);
    double result = 1.0;
    for (int i = 0; i < exp; ++i)
        result *= base;
    return result;
}

// 演示：只有真正使用结果时才触发计算
#define LAZY_POW(base, exp) lazy_pow(base, exp)

// ========== 示例3：延迟初始化的数组（纵：先声明，按需分配页） ==========
typedef struct
{
    int *pages;
    int total_pages;
    int page_size;
} LazyArray;

LazyArray *lazy_array_create(int total_pages, int page_size)
{
    LazyArray *arr = (LazyArray *)malloc(sizeof(LazyArray));
    arr->pages = (int *)calloc(total_pages, sizeof(int)); // 记录哪些页已分配
    arr->total_pages = total_pages;
    arr->page_size = page_size;
    // 实际数据区尚未分配（欲擒故纵）
    return arr;
}

// 访问元素：若所属页未分配，则先分配再访问
int *lazy_array_get(LazyArray *arr, int index)
{
    if (!arr || index < 0)
        return NULL;
    int page_idx = index / arr->page_size;
    if (page_idx >= arr->total_pages)
        return NULL;

    if (arr->pages[page_idx] == 0)
    {
        // 首次访问该页，分配内存（擒：现在才真正分配）
        int *page_data = (int *)calloc(arr->page_size, sizeof(int));
        arr->pages[page_idx] = (intptr_t)page_data; // 强制存储指针
        printf("[延迟分配] 分配第 %d 页 (索引 %d)\n", page_idx, index);
    }
    int *page_data = (int *)arr->pages[page_idx];
    return &page_data[index % arr->page_size];
}

void lazy_array_destroy(LazyArray *arr)
{
    if (arr)
    {
        for (int i = 0; i < arr->total_pages; ++i)
        {
            if (arr->pages[i])
            {
                free((void *)arr->pages[i]);
            }
        }
        free(arr->pages);
        free(arr);
    }
}

// ========== 主函数演示 ==========
int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    srand(time(NULL));

    printf("===== 欲擒故纵：延迟计算与懒加载 =====\n\n");

    // 演示1：懒加载单例
    printf("1. 懒加载对象:\n");
    get_object(); // 第一次，会创建
    get_object(); // 第二次，直接返回
    destroy_object();

    // 演示2：惰性求值
    printf("\n2. 惰性求值宏/函数:\n");
    double x = 5.0;
    printf("调用 LAZY_POW(%.1f, 3) 但未立即使用结果...\n", x);
    double result = LAZY_POW(x, 3); // 此时才真正计算
    printf("计算结果: %.2f\n", result);

    // 演示宏的副作用区别
    int a = 2;
    printf("\nEAGER SQUARE: %d\n", SQUARE_EAGER(a++)); // a 被计算两次，输出 2*2=4, a变成4
    a = 2;
    printf("LAZY SQUARE (GCC): %d\n", LAZY_SQUARE(a++)); // a 只计算一次，输出 2*2=4, a变成3
    // 注意：LAZY_SQUARE 使用了语句表达式（GCC/Clang扩展），若需标准C可改用内联函数

    // 演示3：延迟分配的大数组
    printf("\n3. 延迟分配数组（按需分配页）:\n");
    LazyArray *larr = lazy_array_create(10, 4); // 10页，每页4个int，但未分配任何实际内存
    // 只访问几个元素，只会分配涉及到的页
    *lazy_array_get(larr, 0) = 100;
    *lazy_array_get(larr, 1) = 200;
    *lazy_array_get(larr, 7) =
        300; // 索引7在第1页（0-3是第一页，4-7第二页? page_size=4，索引7属于第1页? 实际页索引=7/4=1）
    *lazy_array_get(larr, 8) = 400; // 索引8属于第2页
    printf("访问元素: larr[0]=%d, larr[7]=%d\n", *lazy_array_get(larr, 0), *lazy_array_get(larr, 7));
    lazy_array_destroy(larr);

    printf("\n欲擒故纵核心思想：\n");
    printf("- 将昂贵操作推迟到必须执行时，提高启动速度\n");
    printf("- 使用静态标志实现单例懒加载\n");
    printf("- 宏或内联函数避免无谓的重复计算\n");
    printf("- 分页/池技术按需分配内存，降低内存占用\n");

    return 0;
}
