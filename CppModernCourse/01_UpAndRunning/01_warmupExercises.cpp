/**
 * @file 01_warmupExercises.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 创建一个名为 absolute_value 的函数, 它返回其参数的绝对值.
 *  取整数x的绝对值的规则如下: 如果x大于或等于0, 则绝对值为x; 否则为x乘以-1
 * 
 * @param value 
 * @return int 
 */
int absolute_value(int value)
{
    if (value < 0)
    {
        return -1 * value;
    }
    else
    {
        return value;
    }
}

/**
 * @brief 函数名为 sum_num 的函数, 该函数接受两个 int 参数并返回它们的和.
 * 
 * @param num1 
 * @param num2 
 * @return int 
 */
int sum_num(const int &num1, const int &num2)
{
    return num1 + num2;
}

// -------------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 感受使用 VSCode + CMake + GDB/LLDB/VS调试器
     * Windows 调试快捷键: F5、F9、F10、F11、Shift+F11;
     * 
     */
    int num1 = -10;
    printf("The absolute value of %d is %d\n", num1, absolute_value(num1));

    int  num2   = 0;
    auto result = absolute_value(num2);
    printf("The absolute value of %d is %d\n", num2, result);

    int  num3    = 42;
    auto result3 = absolute_value(num3);
    printf("The absolute value of %d is %d\n", num3, result3);

    printf("========= Call function sum to add =========\n");
    printf("%d + %d = %d", 42, 12, sum_num(42, 12));

    return 0;
}
