/**
 * @file 18_eliminate_root_problem.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 釜底抽薪: 从根本上解决问题,消除隐患
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 18_eliminate_root_problem.c -o 18_eliminate_root_problem.exe
 *
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// ========== 釜底抽薪1：消除缓冲区溢出 ==========
// 隐患：使用 strcpy、gets、sprintf 等无界函数
// 釜底抽薪：定义安全字符串复制函数（或使用 C11 的 strcpy_s）
void safe_str_copy(char *dest, size_t dest_size, const char *src)
{
    if (!dest || !src || dest_size == 0)
        return;
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

// 釜底抽薪2：消除未初始化变量隐患（强制初始化）
#define SAFE_INIT(var, type, init_val) type var = init_val

// 釜底抽薪3：消除整数溢出隐患（使用检查或更大的类型）
bool safe_add(int a, int b, int *result)
{
    if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b))
    {
        return false; // 溢出
    }
    *result = a + b;
    return true;
}

// 釜底抽薪4：消除动态内存泄漏隐患（使用自动释放，GCC/Clang）
static inline void cleanup_free(void *p)
{
    free(*(void **)p);
}
#define AUTO_FREE __attribute__((cleanup(cleanup_free)))

// 釜底抽薪5：消除空指针解引用隐患（使用 _Nonnull 或断言）
void process_data(const int *ptr)
{
    if (!ptr)
    {
        fprintf(stderr, "错误：空指针，拒绝处理\n");
        return;
    }
    // 安全处理...
}

// ========== 釜底抽薪6：消除条件竞争隐患（使用原子操作） ==========
#include <stdatomic.h>
atomic_int counter = 0;
void increment_safe(void)
{
    atomic_fetch_add(&counter, 1); // 原子操作，无需锁
}

// ========== 釜底抽薪7：消除魔法数字，使用枚举或常量 ==========
typedef enum
{
    MAX_USERS = 100,
    BUFFER_SIZE = 256
} Config;

#define PRINT_VALUE(x) _Generic((x), \
    int: printf("%d", (x)), \
    double: printf("%f", (x)), \
    default: printf("%s", (x)) \
)

// ========== 主函数演示 ==========
int main(int argc, char const *argv[])
{
    SetConsoleOutputCP(65001);

    // 示例1：安全字符串拷贝
    char buffer[10];
    safe_str_copy(buffer, sizeof(buffer), "釜底抽薪示例");
    printf("安全拷贝结果: %s\n", buffer);

    // 示例2：消除未初始化
    SAFE_INIT(value, int, 0);
    printf("初始化的值: %d\n", value);

    // 示例3：安全整数加法
    int result;
    if (safe_add(2000000000, 2000000000, &result))
    {
        printf("加法结果: %d\n", result);
    }
    else
    {
        printf("加法溢出，已阻止\n");
    }

    // 示例4：自动释放内存（釜底抽薪，无需手动 free）
    AUTO_FREE int *p = (int *)malloc(sizeof(int));
    if (p)
    {
        *p = 42;
        printf("自动释放示例: %d (离开作用域自动 free)\n", *p);
    }

    // 示例5：原子操作（无竞争）
    increment_safe();
    printf("原子计数器: %d\n", atomic_load(&counter));

    // 示例6：类型安全打印
    printf("泛型打印: ");
    PRINT_VALUE(123);
    printf(", ");
    PRINT_VALUE(3.14);
    printf(", ");
    PRINT_VALUE("Hello");
    printf("\n");

    printf("\n釜底抽薪核心思想：\n");
    printf("- 使用安全函数替代危险函数（strncpy vs strcpy）\n");
    printf("- 强制初始化、边界检查、溢出检查\n");
    printf("- 利用编译器扩展（cleanup）自动管理资源\n");
    printf("- 原子操作消除锁竞争\n");
    printf("- 类型泛型消除转换隐患\n");
    printf("- 设计上避免问题发生，而非出现问题后再修补\n");

    return 0;
}
