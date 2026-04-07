/**
 * @file 26_payment_gateway.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 假痴不癫: 异常处理与容错设计
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o payment_gateway.exe 26_payment_gateway.c
 *
 */

#include <Windows.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ========== 容错上下文（假痴不癫的伪装） ==========
static jmp_buf recover_env;
static volatile int last_error = 0;

// 错误码定义
typedef enum
{
    ERR_NONE = 0,
    ERR_NETWORK_TIMEOUT,
    ERR_INSUFFICIENT_BALANCE,
    ERR_INVALID_ACCOUNT,
    ERR_SYSTEM_BUSY
} ErrorCode;

// 模拟容错恢复：遇到不可恢复错误时“假装无事”，跳转到安全点
#define TRY                                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (setjmp(recover_env) == 0)                                                                                  \
        {
#define CATCH                                                                                                          \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        handle_error(last_error);                                                                                      \
    }
#define END_TRY                                                                                                        \
    }                                                                                                                  \
    }                                                                                                                  \
    while (0)

// 错误处理函数（假痴不癫：记录但继续）
static void handle_error(ErrorCode err)
{
    switch (err)
    {
    case ERR_NETWORK_TIMEOUT:
        fprintf(stderr, "[容错] 网络超时，稍后重试\n");
        break;
    case ERR_INSUFFICIENT_BALANCE:
        fprintf(stderr, "[容错] 余额不足，使用备用支付方式\n");
        break;
    case ERR_INVALID_ACCOUNT:
        fprintf(stderr, "[容错] 账户无效，请求人工介入\n");
        break;
    default:
        fprintf(stderr, "[容错] 未知错误，继续使用缓存数据\n");
    }
    // 不退出，继续执行
}

// 抛出错误（假痴不癫的“癫”被隐藏）
static void throw_error(ErrorCode err)
{
    last_error = err;
    longjmp(recover_env, 1);
}

// ========== 模拟支付服务（可能失败） ==========
typedef struct
{
    double balance;
    char account[32];
} Account;

// 模拟网络请求（可能超时）
bool deduct_balance(Account *acc, double amount)
{
    // 模拟 50% 概率网络超时
    if (rand() % 2 == 0)
    {
        throw_error(ERR_NETWORK_TIMEOUT);
        return false;
    }
    if (acc->balance < amount)
    {
        throw_error(ERR_INSUFFICIENT_BALANCE);
        return false;
    }
    acc->balance -= amount;
    return true;
}

// 带重试的支付函数（假痴不癫的核心：假装没失败，重试）
bool pay_with_retry(Account *acc, double amount, int max_retries)
{
    for (int i = 0; i < max_retries; ++i)
    {
        // TRY
        {
            if (deduct_balance(acc, amount))
            {
                printf("[支付] 成功扣除 %.2f，余额 %.2f\n", amount, acc->balance);
                return true;
            }
        }
        // CATCH
        {
            // 错误已记录，继续重试
            printf("[重试] 第 %d 次失败，重试中...\n", i + 1);
        }
        // END_TRY
        // 模拟等待
        Sleep(1);
    }
    printf("[支付] 最终失败，使用备用方案\n");
    return false;
}

// 备用支付（假痴不癫的降级）
void fallback_payment(double amount)
{
    printf("[降级] 使用备用支付渠道完成 %.2f 付款\n", amount);
}

// ========== 主程序（假装一切正常） ==========
int main(void)
{

    SetConsoleOutputCP(CP_UTF8);

    srand(time(NULL));

    Account my_acc = {1000.00, "user123"};

    printf("假痴不癫：支付网关容错演示\n");
    printf("初始余额: %.2f\n", my_acc.balance);

    // 尝试支付 500 元（可能遇到网络超时、余额不足等）
    double amount = 500.0;
    bool success = pay_with_retry(&my_acc, amount, 3);

    if (!success)
    {
        fallback_payment(amount);
        // 假痴不癫：即使支付失败，也不退出程序，继续其他业务
    }

    printf("最终余额: %.2f\n", my_acc.balance);
    printf("程序继续执行其他任务...\n");

    // 模拟其他业务不受影响
    printf("其他业务：生成报表...\n");

    return 0;
}
