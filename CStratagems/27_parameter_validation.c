/**
 * @file 27_parameter_validation.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 上屋抽梯: 数据验证和输入边界校验
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 27_parameter_validation.c -o 27_parameter_validation.exe
 * clang 27_parameter_validation.c -o 27_parameter_validation.exe
 *
 */

#include <Windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== 上屋抽梯：安全数组访问 ==========
// 带边界检查的数组读取（抽梯：越界则返回错误）
bool safe_array_get(const int *arr, size_t len, size_t idx, int *out)
{
    if (!arr || !out)
    {
        fprintf(stderr, "[抽梯] 空指针参数\n");
        return false;
    }
    if (idx >= len)
    {
        fprintf(stderr, "[抽梯] 索引 %zu 超出范围 [0, %zu)\n", idx, len);
        return false;
    }
    *out = arr[idx];
    return true;
}

// 带边界检查的数组写入
bool safe_array_set(int *arr, size_t len, size_t idx, int value)
{
    if (!arr)
    {
        fprintf(stderr, "[抽梯] 目标数组为空\n");
        return false;
    }
    if (idx >= len)
    {
        fprintf(stderr, "[抽梯] 写入索引 %zu 超出范围 [0, %zu)\n", idx, len);
        return false;
    }
    arr[idx] = value;
    return true;
}

// ========== 上屋抽梯：整数运算溢出检查 ==========
bool safe_add(int a, int b, int *result)
{
    if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b))
    {
        fprintf(stderr, "[抽梯] 整数溢出: %d + %d\n", a, b);
        return false;
    }
    *result = a + b;
    return true;
}

// ========== 上屋抽梯：字符串长度边界校验 ==========
bool safe_str_copy(char *dest, size_t dest_size, const char *src)
{
    if (!dest || !src)
    {
        fprintf(stderr, "[抽梯] 空指针参数\n");
        return false;
    }
    if (dest_size == 0)
    {
        fprintf(stderr, "[抽梯] 目标缓冲区大小为0\n");
        return false;
    }
    size_t src_len = strlen(src);
    if (src_len >= dest_size)
    {
        fprintf(stderr, "[抽梯] 源字符串长度 %zu 超过缓冲区 %zu\n", src_len, dest_size);
        return false;
    }
    strcpy(dest, src); // 已经确保安全
    return true;
}

// ========== 上屋抽梯：枚举值合法性校验 ==========
typedef enum
{
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_MAX
} Color;

const char *color_to_string(Color c)
{
    static const char *names[] = {"红色", "绿色", "蓝色"};
    if (c >= COLOR_MAX)
    {
        fprintf(stderr, "[抽梯] 非法颜色值 %d\n", c);
        return "未知颜色";
    }
    return names[c];
}

// ========== 使用 _Generic 进行编译时类型检查（上屋抽梯的编译期版本） ==========
#define CHECK_TYPE_INT(x) _Generic((x), int: (x), default: (void)0)
#define SQUARE(x) (CHECK_TYPE_INT(x), (x) * (x))

// ========== 主程序演示 ==========
int main(void)
{
    SetConsoleOutputCP(65001);

    printf("===== 上屋抽梯：数据验证与边界校验 =====\n\n");

    // 示例1：数组越界检查
    int arr[5] = {10, 20, 30, 40, 50};
    int val;
    if (safe_array_get(arr, 5, 2, &val))
        printf("arr[2] = %d\n", val);
    if (!safe_array_get(arr, 5, 10, &val))
        printf("越界访问被拦截\n");

    // 示例2：安全写入
    if (safe_array_set(arr, 5, 4, 99))
        printf("arr[4] 被设置为 99\n");
    safe_array_set(arr, 5, 5, 100); // 失败

    // 示例3：整数溢出检查
    int sum;
    if (safe_add(2000000000, 2000000000, &sum))
        printf("和 = %d\n", sum);
    else
        printf("加法溢出被拦截\n");

    // 示例4：字符串复制
    char buf[10];
    if (safe_str_copy(buf, sizeof(buf), "Hello"))
        printf("复制成功: %s\n", buf);
    if (!safe_str_copy(buf, sizeof(buf), "This string is too long"))
        printf("长字符串复制被拦截\n");

    // 示例5：枚举校验
    printf("颜色: %s\n", color_to_string(COLOR_GREEN));
    printf("非法颜色: %s\n", color_to_string(100));

    // 示例6：编译时类型检查
    int x = 5;
    int sq = SQUARE(x);
    printf("5 的平方 = %d\n", sq);
    // SQUARE(3.14);  // 编译错误，因为类型不匹配

    printf("\n上屋抽梯核心思想：\n");
    printf("- 所有外部输入必须先验证再使用\n");
    printf("- 数组访问前检查索引范围\n");
    printf("- 整数运算前检查溢出可能\n");
    printf("- 字符串复制前检查长度\n");
    printf("- 枚举值检查合法性\n");
    printf("- 使用 _Generic 在编译期捕获类型错误\n");

    return 0;
}
