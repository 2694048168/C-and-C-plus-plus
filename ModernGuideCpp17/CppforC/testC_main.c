/**
 * @file testC_main.c
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "libmathCpp/custom_math_warpC.h"

#include <stdio.h>

// -------------------------------------
int main(int argc, const char **argv)
{
    CustomMathWrapper *obj = CustomMath_create();

    printf("====================================");
    printf("\nThe sum of %d + %d = %d", 24, 12, CustomMath_add(obj, 24, 12));
    printf("\nThe sub of %d - %d = %d", 24, 12, CustomMath_sub(obj, 24, 12));
    printf("\nThe mul of %d * %d = %d", 24, 12, CustomMath_mul(obj, 24, 12));
    printf("\nThe div of %d / %d = %d", 24, 12, CustomMath_div(obj, 24, 12));
    printf("\n====================================\n");

    CustomMath_destroy(obj);

    return 0;
}
