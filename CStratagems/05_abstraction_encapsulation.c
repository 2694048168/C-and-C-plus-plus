/**
 * @file 05_abstraction_encapsulation.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 声东击西: 抽象与封装的艺术
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o abstraction_encapsulation.exe 05_abstraction_encapsulation.c
 * clang -o abstraction_encapsulation.exe 05_abstraction_encapsulation.c
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

//  -----------------------------------------------
// 不透明结构体：用户只能看到指针，无法访问内部
typedef struct Calculator Calculator;

// 支持的运算类型（声东击西：用户选择模式，内部实现可能完全不是字面意思）
typedef enum
{
    MODE_ADD,   // 表面：加法
    MODE_MUL,   // 表面：乘法
    MODE_SECRET // 秘密模式：实际执行减法（声东击西）
} CalcMode;

// 创建计算器实例（封装初始化细节）
Calculator *calc_create(CalcMode mode);

// 销毁计算器
void calc_destroy(Calculator *calc);

// 执行运算：用户以为只是普通计算，实际算法由内部决定
double calc_compute(const Calculator *calc, double a, double b);

// 可选：动态切换策略（运行时“声东击西”）
void calc_switch_mode(Calculator *calc, CalcMode new_mode);
//  -----------------------------------------------

// 定义运算函数类型
typedef double (*ArithmeticFunc)(double, double);

// 具体算法实现（“击西”的真实动作）
static double add_impl(double a, double b)
{
    printf("[内部] 执行加法\n");
    return a + b;
}

static double mul_impl(double a, double b)
{
    printf("[内部] 执行乘法\n");
    return a * b;
}

static double sub_impl(double a, double b)
{
    printf("[内部] 执行减法（秘密模式）\n");
    return a - b;
}

// 声东击西：用户选择 MODE_ADD 时，实际可能做乘法（演示欺骗）
static double fake_add_impl(double a, double b)
{
    printf("[声东击西] 您调用了加法模式，但实际执行乘法！\n");
    return a * b;
}

// 计算器结构体（完全隐藏）
struct Calculator
{
    ArithmeticFunc func; // 当前使用的算法
    CalcMode mode;       // 记录模式（便于调试）
};

Calculator *calc_create(CalcMode mode)
{
    Calculator *calc = (Calculator *)malloc(sizeof(Calculator));
    if (!calc)
        return NULL;
    calc->mode = mode;
    // 根据模式设置实际函数（此处演示“声东击西”）
    switch (mode)
    {
    case MODE_ADD:
        // 正常情况下加法，但我们可以故意用乘法演示欺骗
        // 为了体现“声东击西”，这里使用 fake_add_impl
        calc->func = fake_add_impl; // 用户以为加法，实际乘法
        break;
    case MODE_MUL:
        calc->func = mul_impl;
        break;
    case MODE_SECRET:
        calc->func = sub_impl;
        break;
    default:
        calc->func = add_impl;
        break;
    }
    return calc;
}

void calc_destroy(Calculator *calc)
{
    free(calc);
}

double calc_compute(const Calculator *calc, double a, double b)
{
    if (!calc || !calc->func)
    {
        fprintf(stderr, "计算器无效\n");
        return 0.0;
    }
    // 通过函数指针调用真实实现（用户不知道具体是什么）
    return calc->func(a, b);
}

void calc_switch_mode(Calculator *calc, CalcMode new_mode)
{
    if (!calc)
        return;
    calc->mode = new_mode;
    switch (new_mode)
    {
    case MODE_ADD:
        calc->func = fake_add_impl; // 继续欺骗
        break;
    case MODE_MUL:
        calc->func = mul_impl;
        break;
    case MODE_SECRET:
        calc->func = sub_impl;
        break;
    }
    printf("[声东击西] 模式已切换，但内部算法可能名不副实\n");
}

// ------------------------------
int main(int argc, char const *argv[])
{
    SetConsoleOutputCP(CP_UTF8); // 设置输出代码页为 UTF-8

    // 用户创建一个“加法模式”计算器
    Calculator *calc = calc_create(MODE_ADD);
    if (!calc)
        return 1;

    double x = 6.0, y = 7.0;
    printf("用户调用 compute(%.1f, %.1f)，期望加法得到 13.0\n", x, y);
    double result = calc_compute(calc, x, y);
    printf("实际返回结果: %.1f\n\n", result); // 实际输出 42.0（6*7）

    // 用户切换到乘法模式
    calc_switch_mode(calc, MODE_MUL);
    printf("用户切换到乘法模式，期望 6*7=42\n");
    result = calc_compute(calc, x, y);
    printf("实际结果: %.1f\n\n", result); // 输出 42

    // 秘密模式（减法）
    calc_switch_mode(calc, MODE_SECRET);
    printf("用户切换到秘密模式，期望内部做减法 6-7 = -1\n");
    result = calc_compute(calc, x, y);
    printf("实际结果: %.1f\n", result);

    calc_destroy(calc);
    return 0;
}
