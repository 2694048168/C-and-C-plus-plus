/**
 * @file 00_array_point.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 瞒天过海: 数组访问与指针本质
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 00_array_point.c -o 00_array_point
 * clang 00_array_point.c -o 00_array_point
 *
 */

#include <stdio.h>

int main(void)
{
    int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const size_t arr_len = sizeof(arr) / sizeof(arr[0]);

    for (size_t i = 0; i < arr_len; ++i)
    {
        printf("%d ", *(arr + i));
    }

    printf("\n--------------------------\n");

    for (size_t i = 0; i < arr_len; ++i)
    {
        printf("%d ", arr[i]);
    }

    return 0;
}
