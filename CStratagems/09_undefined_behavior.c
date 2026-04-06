/**
 * @file 09_undefined_behavior.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 笑里藏刀：代码中的陷阱与防御性编程
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 09_undefined_behavior.exe 09_undefined_behavior.c
 * clang -o 09_undefined_behavior.exe 09_undefined_behavior.c
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// ========== 第一部分：笑里藏刀（危险代码） ==========
void dangerous_demo(void)
{
    printf("\n--- 危险演示（请勿模仿）---\n");

    // 陷阱1：缓冲区溢出（经典 gets）
    char small_buf[4];
    // gets(small_buf);          // 危险：无边界检查，若输入超过3字符会溢出
    // 改用模拟输入避免实际崩溃
    strcpy(small_buf, "OVERFLOW");         // 故意溢出，覆盖栈内存
    printf("缓冲区内容: %s\n", small_buf); // 可能崩溃或产生未定义行为

    // 陷阱2：野指针（未初始化指针）
    int *wild_ptr;
    // *wild_ptr = 42;            // 危险：野指针解引用，立刻崩溃或随机修改内存

    // 陷阱3：空指针解引用
    int *null_ptr = NULL;
    if (null_ptr)
    { // 看似检查了，但下面直接使用
        *null_ptr = 10;
    }
    // *null_ptr = 10;            // 危险：空指针解引用

    // 陷阱4：使用已释放的内存
    int *heap_ptr = (int *)malloc(sizeof(int));
    *heap_ptr = 100;
    free(heap_ptr);
    // *heap_ptr = 200;           // 危险：悬垂指针，可能破坏堆管理结构

    // 陷阱5：数组下标越界
    int arr[5] = {0};
    arr[10] = 99; // 越界写，破坏栈内其他变量

    // 陷阱6：有符号整数溢出（未定义行为）
    int max = 2147483647;
    max++; // 溢出，结果不可预测

    printf("危险演示结束（未崩溃只是运气）\n");
}

// ========== 第二部分：防御性编程（化解笑里藏刀） ==========

// 防御1：安全字符串拷贝（使用 strncpy 并手动置零结尾）
void safe_str_copy(char *dest, size_t dest_size, const char *src)
{
    if (!dest || !src || dest_size == 0)
        return;
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

// 防御2：安全分配内存并检查结果
void *safe_malloc(size_t size)
{
    if (size == 0)
        return NULL;
    void *ptr = malloc(size);
    if (!ptr)
    {
        fprintf(stderr, "错误: 内存分配失败 (size=%zu)\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// 防御3：数组访问带边界检查
int safe_array_get(int *arr, size_t len, size_t index)
{
    assert(arr != NULL);
    if (index >= len)
    {
        fprintf(stderr, "错误: 数组越界 index=%zu, len=%zu\n", index, len);
        exit(EXIT_FAILURE);
    }
    return arr[index];
}

// 防御4：使用 _Generic 做编译时类型检查（防止类型错误）
#define CHECK_TYPE_INT(x) _Generic((x), int: 0, default: (void)0)
// 示例：强制要求参数为int
#define SQUARE(x) (CHECK_TYPE_INT(x), (x) * (x))

// 防御5：使用 const 和 restrict 提升可读性和优化
void process_buffer(const int *restrict src, int *restrict dst, size_t n)
{
    if (!src || !dst)
        return;
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = src[i] * 2;
    }
}

// 防御6：使用静态断言检查编译期常量
static_assert(sizeof(int) >= 4, "int 必须至少4字节，否则代码需要调整");

// 防御7：安全输入读取（fgets + 移除换行）
bool safe_read_line(char *buffer, size_t size, FILE *stream)
{
    if (!buffer || size == 0 || !stream)
        return false;
    if (fgets(buffer, (int)size, stream) == NULL)
        return false;
    // 移除末尾换行符
    buffer[strcspn(buffer, "\n")] = '\0';
    return true;
}

// 防御8：模拟资源分配与释放（防止泄漏）
typedef struct
{
    int *data;
    size_t len;
} SafeArray;

SafeArray *safe_array_create(size_t len)
{
    SafeArray *arr = (SafeArray *)malloc(sizeof(SafeArray));
    if (!arr)
        return NULL;
    arr->data = (int *)calloc(len, sizeof(int));
    if (!arr->data)
    {
        free(arr);
        return NULL;
    }
    arr->len = len;
    return arr;
}

void safe_array_destroy(SafeArray *arr)
{
    if (arr)
    {
        free(arr->data);
        free(arr);
    }
}

// 防御9：使用枚举代替魔数
typedef enum
{
    STATUS_OK = 0,
    STATUS_NULL_PTR,
    STATUS_OUT_OF_RANGE,
    STATUS_ALLOC_FAIL
} Status;

Status divide_safe(int a, int b, int *result)
{
    if (!result)
        return STATUS_NULL_PTR;
    if (b == 0)
        return STATUS_OUT_OF_RANGE;
    *result = a / b;
    return STATUS_OK;
}

// 防御10：使用 assert 捕捉不可能发生的条件（仅在调试模式）
void tricky_function(int *ptr)
{
    assert(ptr != NULL); // 调试时立即暴露问题
    *ptr = 42;
}

// ========== 主函数：对比演示 ==========
int main(int argc, char const *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    // 不运行危险代码（避免真正崩溃），只注释说明
    // dangerous_demo();   // 取消注释可能崩溃

    printf("===== 防御性编程演示 =====\n");

    // 演示安全字符串拷贝
    char buf[10];
    safe_str_copy(buf, sizeof(buf), "HelloWorldLong");
    printf("安全拷贝结果: '%s' (长度最多9)\n", buf);

    // 演示安全数组访问
    int arr[] = {10, 20, 30};
    int val = safe_array_get(arr, 3, 1);
    printf("安全数组访问 arr[1] = %d\n", val);
    // safe_array_get(arr, 3, 5);  // 会触发错误并退出

    // 演示类型检查宏
    int x = 5;
    int sq = SQUARE(x); // 正确
    // SQUARE(3.14);      // 编译错误，因为参数不是int

    // 演示安全除法
    int res;
    Status st = divide_safe(10, 2, &res);
    if (st == STATUS_OK)
        printf("10/2 = %d\n", res);
    st = divide_safe(10, 0, &res);
    if (st == STATUS_OUT_OF_RANGE)
        printf("除零错误被捕获\n");

    // 演示安全数组创建与销毁
    SafeArray *my_arr = safe_array_create(5);
    if (my_arr)
    {
        my_arr->data[0] = 100;
        printf("安全数组第一个元素: %d\n", my_arr->data[0]);
        safe_array_destroy(my_arr);
    }

    // 演示断言（仅调试模式有效）
    int num = 123;
    tricky_function(&num);
    // tricky_function(NULL);   // 断言失败，程序终止（调试时）

    printf("\n防御性编程核心：\n");
    printf("- 所有外部输入必须验证边界\n");
    printf("- 指针使用前检查非空\n");
    printf("- 资源分配后检查成功并配对释放\n");
    printf("- 使用安全函数（strncpy, snprintf, fgets）\n");
    printf("- 开启编译器警告（-Wall -Wextra）\n");
    printf("- 使用静态分析工具（clang-analyzer, cppcheck）\n");

    return 0;
}
