/**
 * @file 01_function_callback.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 围魏救赵: 函数指针与回调函数
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 01_function_callback 01_function_callback.c
 * clang -o 01_function_callback 01_function_callback.c
 *
 */

#include <stdio.h>

// 定义回调函数类型：无参数、无返回值的函数指针
typedef void (*SiegeWeiStrategy)(void);

// 主控函数：执行“围魏”回调，然后完成“救赵”
void rescue_zhao(SiegeWeiStrategy besiege_wei)
{
    if (besiege_wei != NULL)
    {
        printf("[rescue_zhao] Begin SiegeWeiStrategy...\n");
        besiege_wei(); // 调用回调
        printf("[rescue_zhao] rescue_zhao Successfully!\n");
    }
    else
    {
        printf("[rescue_zhao] rescue_zhao NOT Successfully.\n");
    }
}

// 具体策略1：火攻魏国粮草
void fire_attack(void)
{
    printf("   [SiegeWeiStrategy] The Wei army was cut off from food and supplies after their grain and forage depot "
           "was attacked by fire.\n");
}

// 具体策略2：截断魏国补给线
void cut_supply_line(void)
{
    printf("   [SiegeWeiStrategy] Dispatch cavalry to cut off the grain supply route of the State of Wei.\n");
}

int main(int argc, char const *argv[])
{
    // 使用不同回调策略间接实现救援
    rescue_zhao(fire_attack);
    printf("\n");

    rescue_zhao(cut_supply_line);
    printf("\n");

    rescue_zhao(NULL); // 无策略演示

    return 0;
}
