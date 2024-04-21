/**
 * @file 01_fundamentalFloatTypes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief Floating-Point Types 浮点类型存储的是实数,
 * (定义为任何带有小数点和小数部分的数字，如0.33333或98.6)的近似值;
 * 虽然无法在计算机内存中准确地表示某些实数, 但可以存储一个近似值;
 * 如果这看起来很难相信, 那么可以想一想像π这样的数字, 它有无限多的位数;
 *  在有限的计算机内存中, 怎么可能表示无限多位的数字?
 * 与所有其他类型一样, 浮点类型占用的内存是有限的, 这被称为类型的精度,
 * 浮点类型的精度越它对实数的近似就越准确, C++为近似值提供了三个级别的精度:
 * 1. float: 单精度;
 * 2. double: 双精度;
 * 3. long double: 扩展精度;
 * 
 * 和整数类型一样, 每种浮点表示都取决于实现, 请注意, 这些实现方式存在大量的细微差别
 * NOTE: 注意 对于那些不能安全地忽略浮点表示细节的人来说, 
 *   可以看看与自己硬件平台相关的浮点规范。
 *   浮点存储和算术的主要实现方式在《IEEE浮点算术标准》(IEEE 754)
 * 
 */
// -----------------------------------
int main(int argc, const char **argv)
{
    // ======== 浮点字面量 Floating-Point Literals ========
    /**
     * @brief 浮点字面量 Floating-Point Literals
     * 浮点字面量默认为双精度, 如果需要单精度字面量, 则使用f或F后缀;
     * 如果需要扩展精度字面则使用|或L;
     * 字面量也可以使用科学计数法:
     * constant = 6.626070040e-342
     * 基数和指数之间不允许有空格.
     * 
     * 2.浮点格式指定符
     * 格式指定符 %f 显示带有小数位的浮点数,
     * 而 %e 则以科学计数法显示相同的数字,
     * 也可以让 printf 使用 %g 格式指定符, 选择 %e 或 %f 中更紧凑的一个;
     * 对于 double, 只需在说明符前面加上小写字母l,
     * 而对于 long double, 在前面加上大写字母L;
     * 一般来说，使用 %g 来打印浮点类型,
     * NOTE: 在实践中, 可以省略 double 格式指定符中的l前缀, 
     *      因为 printf 会将浮点数参数提升为双精度类型.
     * 
     */
    float num_a = 0.1f;
    printf("the single precision value: %f\n", num_a);

    double num_b = 0.2;
    printf("the double precision value: %f\n", num_b);
    printf("the double precision value: %lf\n", num_b);

    long double num_c = 3.14;
    printf("the double precision value: %Lf\n", num_c);

    double an = 6.0221409e23;
    printf("Avogadro's Number: %le %lf %lg\n", an, an, an);
    float hp = 9.75;
    printf("Hogwarts' Platform: %e %f %g\n", hp, hp, hp);

    return 0;
}
